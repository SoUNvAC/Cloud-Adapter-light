_base_ = ["./phase22_clean_v8_l1c.py"]

import os

pretrained = os.environ.get(
    "PHASE26_RESNET18_PRETRAINED",
    "work_dirs/phase26_resnet18_pilot/pretrained/resnet18_v1c-b5776b93.pth",
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="ResNetV1c",
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        strides=[4, 8, 16, 32],
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={"backbone": dict(lr_mult=0.1), "norm": dict(decay_mult=0.0)},
        norm_decay_mult=0.0,
    ),
)

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = val_dataloader

work_dir = "./work_dirs/phase26_resnet18_pilot/seed42"
