import torch
import math
import numpy as np
import torch.nn as nn

from mmcv.runner import load_checkpoint
from mmaction.utils import get_root_logger
from ..builder import BACKBONES

from timm.models.layers import DropPath, trunc_normal_


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

class LWAttention(nn.Module):
    """
    LW-MSA: Local Window-based MSA
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., local_size=(1, 1, 1)):
        super(LWAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.local_size = local_size

    def forward(self, x, D, H, W):
        # x: shape (B, N, C), B is the batch size, N is the number of tokens, C is the channel number
        # D, H, W: resolution of input feature map
        B, N, C = x.shape

        nd, nh, nw = D // self.local_size[0], H // self.local_size[1], W // self.local_size[2]
        nl = nd * nh * nw  # the number of local windows

        x = x.reshape(B, nd, self.local_size[0], nh, self.local_size[1],
                      nw, self.local_size[2], C).permute(0, 1, 3, 5, 2, 4, 6, 7)

        qkv = self.qkv(x).reshape(B, nl, -1, 3, self.num_heads,
                                  C // self.num_heads).permute(3, 0, 1, 4, 2, 5)

        # B, hw, n_head, prod(local_size), head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        # B, hw, n_head, prod(local_size), prod(local_size)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(
            B, nd, nh, nw, self.local_size[0], self.local_size[1], self.local_size[2], C)
        x = attn.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GPAttention(nn.Module):
    """
    GP-MSA: Global Pyramid-based MSA.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 fine_pysize=(8, 7, 7), coarse_pysize=(4, 4, 4), resolution=(16, 56, 56), stage=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.resolution = resolution
        self.stage = stage

        self.fine_pysize = fine_pysize
        self.fine_kernel_size = [rs // ts for rs,
                            ts in zip(resolution, fine_pysize)]
        if np.prod(self.fine_kernel_size) > 1:
            # Fine-grained level pyramid
            self.sr = nn.Conv3d(
                dim, dim, kernel_size=self.fine_kernel_size, stride=self.fine_kernel_size, groups=dim)
            self.norm = nn.LayerNorm(dim)

            # Coarse-grained level pyramid, only used in the first two stages
            # We factorize the 3d conv in pyramid downsampling to 2D+1D conv (spatially and temporally separable)
            self.coarse_pysize = coarse_pysize
            self.coarse_kernel_size = [rs // ts \
                for rs, ts in zip(resolution, coarse_pysize)]
            lr_kernel1 = [1, 7, 7]
            lr_kernel2 = [rs // ts \
                for rs, ts in zip(self.coarse_kernel_size, lr_kernel1)]
            if self.stage == 0:
                # Deprecated implementation since using conv3d with kernel 1,14,14 and 4,1,1 is very slow
                # self.lr = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=(1, 14, 14), stride=(1, 14, 14), groups=dim),
                #                         nn.BatchNorm3d(dim),
                #                         nn.ReLU(inplace=True),
                #                         nn.Conv3d(dim, dim, kernel_size=(4, 1, 1), stride=(4, 1, 1), groups=dim))
                #
                # Current implementation: in practice, using kernel 1,7,7 and 4,2,2 is ~30% faster than the above method
                self.lr = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=lr_kernel1, stride=lr_kernel1, groups=dim),
                                        nn.BatchNorm3d(dim),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(dim, dim, kernel_size=lr_kernel2, stride=lr_kernel2, groups=dim))
            elif self.stage == 1:
                self.lr = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=lr_kernel1, stride=lr_kernel1, groups=dim),
                                        nn.BatchNorm3d(dim),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(dim, dim, kernel_size=lr_kernel2, stride=lr_kernel2, groups=dim))

    def forward(self, x, D, H, W):
        # x: shape (B, N, C), B is the batch size, N is the number of tokens, C is the channel number
        # D, H, W: resolution of input feature map
        B, N, C = x.shape
        q = self.q(x).reshape(
            B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if np.prod(self.fine_kernel_size) > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            if self.stage < 2:  # two levels of pyramids
                x__ = self.lr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = torch.cat([x_, x__], dim=1)
            else:  # only the fine-grained level
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //
                                    self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DualFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, resolution=(16, 7, 7), mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fine_pysize=(8, 7, 7), coarse_pysize=(4, 4, 4), local_size=(1, 1, 1), stage=0):
        super().__init__()
        if local_size == (1, 1, 1):
            self.attn = GPAttention(dim, num_heads, qkv_bias,
                                    qk_scale, attn_drop, drop,
                                    fine_pysize, coarse_pysize, resolution, stage)
        else:
            self.attn = LWAttention(dim, num_heads, qkv_bias,
                                    qk_scale, attn_drop, drop, local_size)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, D, H, W):
        b = x.shape[0]
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # This reshape is for gradcam visualization
        x = x.reshape(b, D, H, W, -1).permute(0, 4, 1, 2, 3)  
        return x

# Borrow from PVT and Twins-SVT
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, video_size=(32, 224, 224), patch_size=(2, 4, 4), in_chans=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        assert video_size[0] % patch_size[0] == 0 and video_size[1] % patch_size[1] == 0, \
            f"video_size {video_size} should be divided by patch_size {patch_size}."
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: shape [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)  # [B, N, C]
        D, H, W = D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        return x, (D, H, W)

class PosCNN(nn.Module):
    # PEG from https://arxiv.org/abs/2102.10882
    # We change the original conv to depth-wise conv for improving efficiency
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, D, H, W):
        # x: shape (B, N, C), B is the batch size, N is the number of tokens, C is the channel number
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, D, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x  # [B, N, C]

# The code template is built on three transformers, including PVT, Twins and Swin.
# We really appreciate the code template provided by them.
# PVT: https://github.com/whai362/PVT.git
# Twins: https://github.com/Meituan-AutoML/Twins
# Swin Transformer: https://github.com/SwinTransformer/Video-Swin-Transformer
@BACKBONES.register_module()
class DualFormer(nn.Module):
    def __init__(self,
                 pretrained=None,  # pretrained model path.
                 pretrained2d=True, # True if the pretrained model is trained on image datasets.
                 video_size=(32, 224, 224), # Input video size, RGB not included.
                 patch_size=(2, 4, 4),  # The 3D patch size.
                 in_chans=3,  # Input channels. Default: RGB - 3.
                 num_classes=1000,  # The number of classes for recognition.
                 embed_dims=[64, 128, 256, 512], # Hidden dimensionality in different stages.
                 num_heads=[1, 2, 4, 8], # The number of heads in different stages.
                 mlp_ratios=[4, 4, 4, 4], # The MLP expansion rate in different stages.
                 qkv_bias=False,  # Whether adding bias to qkv.
                 qk_scale=None,  # Whether scaling on qk.
                 drop_rate=0.,  # Dropout rate.
                 attn_drop_rate=0.,  # Dropout rate on attention values.
                 drop_path_rate=0.,  # Drop path rate.
                 norm_layer=nn.LayerNorm,  # The norm layer.
                 depths=[2, 2, 10, 4],  # The number of blocks in each stage.
                 local_sizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7)], # local window size
                 fine_pysizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7), (8, 7, 7)], # Fine-grained pyramid size
                 coarse_pysizes=[(4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 4, 4)], # Coarse-grained pyramid size
                 temporal_pooling=[-1, 1, 1, 1]): # temporal pooling rate in pos_embed
        super().__init__()
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.local_sizes = local_sizes
        self.fine_pysizes = fine_pysizes
        temporal_pooling[0] = patch_size[0]
        self.temporal_pooling = temporal_pooling

        temporal_pooling_rate = 1
        vs = video_size
        self.nvs = []
        spatial_pooling = (4, 2, 2, 2)
        for i in range(len(depths)):
            vs = (vs[0] // temporal_pooling[i], vs[1] //
                  spatial_pooling[i], vs[2] // spatial_pooling[i])
            self.nvs.append(vs)

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(
                    video_size, patch_size, in_chans, embed_dims[i]))
            else:
                new_video_size = (
                    video_size[0] // temporal_pooling_rate,
                    video_size[1] // patch_size[1] // 2 ** (i - 1),
                    video_size[2] // patch_size[2] // 2 ** (i - 1))
                self.patch_embeds.append(PatchEmbed(new_video_size,
                                                    (self.temporal_pooling[i], 2, 2),
                                                    embed_dims[i - 1],
                                                    embed_dims[i]))
                self.nvs.append(new_video_size)

            self.pos_drops.append(nn.Dropout(p=drop_rate))
            temporal_pooling_rate *= self.temporal_pooling[i]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([DualFormerBlock(dim=embed_dims[k],
                                                    num_heads=num_heads[k],
                                                    mlp_ratio=mlp_ratios[k],
                                                    qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=dpr[cur + i],
                                                    norm_layer=norm_layer,
                                                    fine_pysize=fine_pysizes[k],
                                                    coarse_pysize=coarse_pysizes[k],
                                                    resolution=self.nvs[k],
                                                    local_size=(1, 1, 1) \
                                                        if i % 2 == 1 else local_sizes[k],
                                                    stage=k)
                                    for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm = norm_layer(embed_dims[-1])

        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )

        # self.init_weights(pretrained) # done by mmaction2, do not need this line

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained,
                                strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def inflate_weights(self, logger=None):
        """
        Inflate the 2d parameters to 3d. 
        """
        state_dict = torch.load(self.pretrained, map_location='cpu')

        # Inflate patch embedding layer
        patch_embed_weight_keys = [
            k for k in state_dict.keys() if "patch_embeds" in k and "proj.weight" in k]
        for i, k in enumerate(patch_embed_weight_keys):
            if i >= len(self.depths):
                continue
            state_dict[k] = state_dict[k].unsqueeze(2).repeat(
                1, 1, self.temporal_pooling[i], 1, 1) / self.temporal_pooling[i]

        # Inflate the PEG
        pos_block_keys = [k for k in state_dict.keys(
        ) if "pos_block" in k and "weight" in k]
        for i, k in enumerate(pos_block_keys):
            if i >= len(self.depths):
                continue
            state_dict[k] = state_dict[k].unsqueeze(2).repeat(
                1, 1, 3, 1, 1) / 3

        # Inflate the fine-grained level of each GP-MSA layer
        sr_proj_keys = [k for k in state_dict.keys() if "sr.weight" in k]
        for i, k in enumerate(sr_proj_keys):
            if int(k[7]) >= len(self.depths):
                continue
            expansion = self.nvs[int(k[7])][0] // self.fine_pysizes[int(k[7])][0]
            state_dict[k] = state_dict[k].unsqueeze(2).repeat(
                1, 1, expansion, 1, 1) / expansion

        # Inflate the coarse-grained level of each GP-MSA layer
        lr_keys = [k for k in state_dict.keys() if "attn.lr.0.weight" in k]
        for i, k in enumerate(lr_keys):
            state_dict[k] = state_dict[k].unsqueeze(2)

        lr_keys = [k for k in self.state_dict().keys()
                   if "attn.lr.3.weight" in k]
        for i, k in enumerate(lr_keys):
            temp = k.split('.')
            if int(temp[1]) == 0: # stage 0
                state_dict[k] = torch.ones_like(self.state_dict()[k]) / (4*2*2) # divided by kernel size
            elif int(temp[1]) == 1: # stage 1
                state_dict[k] = torch.ones_like(self.state_dict()[k]) / (4*1*1) # divided by kernel size
            else:
                print(temp) # should be rewritten when using coarse-grained levels at stage 2 or 3

        lr_keys = [k for k in self.state_dict().keys()
                   if "attn.lr.3.bias" in k]
        for i, k in enumerate(lr_keys):
            state_dict[k] = torch.zeros_like(self.state_dict()[k]) # bias is initialized by 0

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del state_dict
        torch.cuda.empty_cache()

    def forward_features(self, x):
        # x: [b, c, d, h, w]
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (D, H, W) = self.patch_embeds[i](x)  # [B, N, C], (D, H, W)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, D, H, W)
                x = x.reshape(B, -1, D*H*W).permute(0, 2, 1) # for gradcam
                if j == 0:
                    x = self.pos_block[i](x, D, H, W)  # PEG here
            x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)

        x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x  # [B, C, D, H, W]

    def forward(self, x):
        x = self.forward_features(x)
        return x

# For testing
# def DualFormer_Tiny(video_size):
#     return DualFormer(
#         pretrained='checkpoints/pretrained_2d/dualformer_tiny_new.pth',
#         pretrained2d=True,
#         video_size=video_size,
#         patch_size=(2, 4, 4),
#         in_chans=3,
#         num_classes=1000,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.1,
#         embed_dims=[64, 128, 256, 512],
#         num_heads=[2, 4, 8, 16],
#         mlp_ratios=[4, 4, 4, 4],
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=[2, 2, 10, 4],
#         local_sizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7), (8, 7, 7)],
#         fine_pysizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7), (16, 7, 7)],
#         temporal_pooling=[-1, 1, 1, 1],
#     )
# if __name__ == '__main__':
#     x = torch.rand(1, 3, 32, 224, 224).fill_(0)
#     model = DualFormer_Tiny(x.shape[2:])
#     y = model(x)
#     from fvcore.nn import FlopCountAnalysis, parameter_count_table
#     flops = FlopCountAnalysis(model, x)
#     print('flops: {:.3f} G'.format(flops.total() / 1e9))
#     counter = flops.by_module().most_common()
#     # for c in counter:
#     #     print('{} FLOPs: {} G'.format(c[0], c[1] / 1e9))
#     print('Params: {}'.format(parameter_count_table(model)))
#     print(y.shape)
