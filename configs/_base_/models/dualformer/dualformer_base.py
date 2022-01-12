_base_ = "dualformer_tiny.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2],
                           embed_dims=[128, 256, 512, 1024],
                           num_heads=[4, 8, 16, 32]),
             cls_head=dict(in_channels=1024))
