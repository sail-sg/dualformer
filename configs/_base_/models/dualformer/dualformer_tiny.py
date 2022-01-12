# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='DualFormer',
        video_size=(32, 224, 224),
        patch_size=(2, 4, 4),
        in_chans=3,
        num_classes=1000,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 10, 4],
        temporal_pooling=[-1, 1, 1, 1],
        local_sizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7), (8, 7, 7)],
        fine_pysizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7), (16, 7, 7)]),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob'))
