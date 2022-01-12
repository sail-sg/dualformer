_base_ = "dualformer_tiny.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2],
                           embed_dims=[96, 192, 384, 768],
                           num_heads=[3, 6, 12, 24]),
             cls_head=dict(in_channels=768))
