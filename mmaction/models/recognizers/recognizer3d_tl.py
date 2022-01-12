import torch
from torch import nn
import numpy as np

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from ..backbones import DualFormer
from ..heads.i3d_head import I3DHead_Teacher


class Teacher(nn.Module):
    def __init__(self, backbone, cls_head):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        return x


def load_teacher_model(path='checkpoints/k400/dualformer_base_patch244_window877.pth',
                       last_embed_dim=1024,
                       output_dim=400,
                       info=False):
    # The teacher model is employed on the fly. We will implement the offline version later.
    # Args:
    #    path: annotation model path
    #    last_embed_dim: hidden dimensionality at the last stage
    #    output_dim: the number of classes, e.g., in K400 this value is 400

    state_dict = torch.load(path, map_location='cpu')

    # Config of DualFormer-B
    backbone = DualFormer(pretrained=path,
                          pretrained2d=True,
                          video_size=(32, 224, 224),
                          patch_size=(2, 4, 4),
                          in_chans=3,
                          num_classes=1000,
                          qk_scale=None,
                          drop_rate=0.,
                          attn_drop_rate=0.,
                          drop_path_rate=0.,
                          embed_dims=[128, 256, 512, 1024],
                          num_heads=[4, 8, 16, 32],
                          mlp_ratios=[4, 4, 4, 4],
                          qkv_bias=True,
                          depths=[2, 2, 18, 2],
                          temporal_pooling=[-1, 1, 1, 1],
                          local_sizes=[(8, 7, 7), (8, 7, 7),
                                       (8, 7, 7), (8, 7, 7)],
                          fine_pysizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7), (16, 7, 7)])

    cls_head = I3DHead_Teacher(output_dim, last_embed_dim, dropout_ratio=0.5)

    teacher = Teacher(backbone, cls_head)
    msg = teacher.load_state_dict(state_dict, strict=False)
    if info:
        print(msg)

    return teacher


@ RECOGNIZERS.register_module()
class Recognizer3D_TL(BaseRecognizer):
    """3D recognizer model framework."""

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Recognizer3D_TL, self).__init__(backbone,
                                              cls_head,
                                              neck,
                                              train_cfg,
                                              test_cfg)

        self.teacher = load_teacher_model().cuda()  # load teacher model
        self.teacher.eval()  # set eval mode
        for param in self.teacher.parameters():  # turn off gradients
            param.requires_grad = False

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        # [N, num_classes], [N, D, 7, 7, n_classes]
        cls_score, token_preds = self.cls_head(x)

        # using teacher model
        teacher_preds = self.teacher(imgs)  # [N, 16, 7, 7, n_classes]
        B, D, H, W, C = teacher_preds.shape

        # If inconsistent shapes occur (generally happen when using different teacher and student models),
        # we average the predictions to obtain consistent sizes of representations.
        if token_preds.shape[1] != teacher_preds.shape[1]:
            ratio = teacher_preds.shape[1] // token_preds.shape[1]
            teacher_preds = teacher_preds.reshape(B, ratio, D//ratio, H, W, C)
            teacher_preds = teacher_preds.mean(dim=1)

        gt_labels = labels.squeeze()

        loss_cls = self.cls_head.loss(cls_score, gt_labels,
                                      token_preds=token_preds.reshape(-1, C),
                                      teacher_preds=teacher_preds.reshape(
                                          -1, C),
                                      gamma=0.5)

        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat, tuple):
                feat = [avg_pool(x) for x in feat]
                # concat them
                feat = torch.cat(feat, axis=1)
            else:
                feat = avg_pool(feat)
            # squeeze dimensions
            feat = feat.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score, _ = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs, _ = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
