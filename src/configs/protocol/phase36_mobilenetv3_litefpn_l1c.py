_base_ = ["./phase31_mobilenetv2_litefpn_l1c.py"]

import os

pretrained = os.environ.get(
    "PHASE36_MOBILENETV3_PRETRAINED",
    "work_dirs/phase36_mobilenetv3_litefpn/pretrained/"
    "mobilenet_v3_large_imagenet1k_v2_features.pth",
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="TorchvisionMobileNetV3Large",
        out_indices=(3, 6, 12, 15),
        norm_eval=False,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    decode_head=dict(
        in_channels=[24, 40, 112, 160],
        strides=[4, 8, 16, 32],
    ),
)

work_dir = "./work_dirs/phase36_mobilenetv3_litefpn/seed42"
