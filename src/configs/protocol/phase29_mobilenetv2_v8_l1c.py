_base_ = ["./phase26_resnet18_v8_l1c.py"]

import os

pretrained = os.environ.get(
    "PHASE29_MOBILENETV2_PRETRAINED",
    "work_dirs/phase29_mobilenetv2_pilot/pretrained/"
    "mobilenet_v2_backbone_only.pth",
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="MobileNetV2",
        widen_factor=1.0,
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    decode_head=dict(
        in_channels=[24, 32, 96, 320],
        strides=[4, 8, 16, 32],
    ),
)

work_dir = "./work_dirs/phase29_mobilenetv2_pilot/seed42"
