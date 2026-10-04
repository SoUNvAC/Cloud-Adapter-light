"""Source parent-space readout used by every Phase 64 target method."""

_base_ = ["./phase22_clean_v8_l1c.py"]

import os


parent_classes = ("surface_visible", "cloud", "shadow")
parent_palette = [[80, 130, 70], [245, 245, 245], [70, 70, 70]]
parent_metainfo = dict(classes=parent_classes, palette=parent_palette)
source_root = os.environ.get(
    "PHASE64_CLOUDSEN_ROOT", "data/cloudsen12_high_l1c"
)

model = dict(
    type="FrozenBackboneEncoderDecoder",
    decode_head=dict(
        num_classes=3,
        loss_cls=dict(class_weight=[1.0, 1.0, 1.0, 0.1]),
    ),
)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type="Phase64CloudSENParentDataset",
        data_root=source_root,
        metainfo=parent_metainfo,
    ),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type="Phase64CloudSENParentDataset",
        data_root=source_root,
        metainfo=parent_metainfo,
    ),
)
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type="Phase64CloudSENParentDataset",
        data_root=source_root,
        metainfo=parent_metainfo,
    ),
)

load_from = (
    "work_dirs/phase64_source_parent_init/source_without_native_classifier.pth"
)
optim_wrapper = dict(optimizer=dict(lr=1e-4, weight_decay=0.05))
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=100),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=100, end=4000, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=4000, val_interval=500)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=500,
        max_keep_ckpts=2,
        save_best="mIoU",
        rule="greater",
    ),
)
randomness = dict(seed=64, deterministic=True)
work_dir = "./work_dirs/phase64_source_parent/seed64"
