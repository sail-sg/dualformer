# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, num_frame, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.num_frame = num_frame
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            T_sp, H_sp, W_sp = num_frame, resolution, resolution
        elif idx == 0:
            T_sp, H_sp, W_sp = num_frame, resolution, split_size
        elif idx == 1:
            T_sp, H_sp, W_sp = num_frame, split_size, resolution
        elif idx == 2:
            T_sp, H_sp, W_sp = split_size, resolution, resolution
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.T_sp = T_sp
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv3d(dim, dim, kernel_size=3,
                               stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, L, C = x.shape
        T, H, W = self.num_frame, self.resolution, self.resolution
        assert L == T * H * W
        x = x.transpose(-2, -1).contiguous().view(B, C, T, H, W)
        x = img2windows(x, self.T_sp, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.T_sp*self.H_sp*self.W_sp, self.num_heads,
                      C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, L, C = x.shape
        T, H, W = self.num_frame, self.resolution, self.resolution
        assert L == T * H * W
        x = x.transpose(-2, -1).contiguous().view(B, C, T, H, W)

        T_sp, H_sp, W_sp = self.T_sp, self.H_sp, self.W_sp
        x = x.view(B, C, T // T_sp, T_sp, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous(
        ).reshape(-1, C, T_sp, H_sp, W_sp)

        lepe = func(x)  # B', C, T', H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads,
                            T_sp * H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.T_sp *
                      self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Img2Window
        T, H, W = self.num_frame, self.resolution, self.resolution
        B, L, C = q.shape
        assert L == T * H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)

        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        # B head N C @ B head C N --> B head N N
        attn = (q @ k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.T_sp*self.H_sp *
                                      self.W_sp, C)  # B head N N @ B head N C

        # Window2Img
        x = windows2img(x, self.T_sp, self.H_sp, self.W_sp, T,
                        H, W).view(B, -1, C)  # B T' H' W' C

        return x


class CSWinBlock(nn.Module):
    def __init__(self, dim, num_frame, reso, num_heads,
                 split_size=(8, 7, 7), mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frame = num_frame
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 3
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, num_frame=num_frame, resolution=self.patches_resolution, idx=-1,
                    split_size=0, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//3, num_frame=num_frame, resolution=self.patches_resolution, idx=i,
                    split_size=split_size[i], num_heads=num_heads//3, dim_out=dim//3,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, T*H*W, C
        """

        H = W = self.patches_resolution
        T = self.num_frame
        B, L, C = x.shape
        assert L == T * H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 3:
            x1 = self.attns[0](qkv[:, :, :, :C//3])
            x2 = self.attns[1](qkv[:, :, :, C//3:-C//3])
            x3 = self.attns[2](qkv[:, :, :, -C//3:])
            attened_x = torch.cat([x1, x2, x3], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, T_sp, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, T, H, W = img.shape
    img_reshape = img.view(B, C, T // T_sp, T_sp, H //
                           H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(
        0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, T_sp*H_sp*W_sp, C)
    return img_perm


def windows2img(img_splits_hw, T_sp, H_sp, W_sp, T, H, W):
    """
    img_splits_hw: B' T H W C
    """
    B = int(img_splits_hw.shape[0] / (T * H * W // T_sp // H_sp // W_sp))

    img = img_splits_hw.view(B, T // T_sp, H // H_sp,
                             W // W_sp, T_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.norm = norm_layer(dim_out)

    def forward(self, x, T=16):
        B, new_THW, C = x.shape
        H = W = int(np.sqrt(new_THW // T))
        x = x.transpose(-2, -1).contiguous().view(B, C, T, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, num_frame=32, img_size=224, patch_size=(2, 4, 4), in_chans=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2], split_size=[3, 5, 7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_frame = num_frame
        heads = num_heads

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, (2, 7, 7), (2, 4, 4), (0, 2, 2)),
            Rearrange('b c t h w -> b (t h w) c', t=num_frame //
                      patch_size[0], h=img_size//patch_size[1], w=img_size//patch_size[2]),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], num_frame=num_frame//2, reso=img_size//4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], num_frame=num_frame//2, reso=img_size//8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer)
             for i in range(depth[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], num_frame=num_frame//2, reso=img_size//16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer)
             for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.merge3 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], num_frame=num_frame//2, reso=img_size//32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True)
             for i in range(depth[-1])])

        self.norm = norm_layer(curr_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            print('='*20)
            print(x.shape)
            x = pre(x, self.num_frame//2)
            print(x.shape)
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        x = self.norm(x)
        return torch.mean(x, dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 32, 224, 224).fill_(0)
    model = CSWinTransformer(num_frame=32,
                             img_size=224,
                             patch_size=(2, 4, 4),
                             in_chans=3,
                             embed_dim=96,
                             depth=[1, 2, 21, 1],
                             split_size=[(1, 1, 1), (2, 2, 2), (7, 7, 4), (7, 7, 8)],
                             num_heads=[3, 6, 9, 18],
                             mlp_ratio=4.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_rate=0.,
                             attn_drop_rate=0.,
                             drop_path_rate=0.,
                             hybrid_backbone=None,
                             norm_layer=nn.LayerNorm,
                             use_chk=False)
    y = model(x)
    print(y.shape)
