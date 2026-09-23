_base_ = ["./phase22_clean_v8_l1c.py"]

import os

manifest_path = "work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv"
selection_path = "work_dirs/phase50_active_1pct/selection.json"
source_root = os.environ.get("CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c")
source_metainfo = dict(
    classes=("clear", "thick cloud", "thin cloud", "cloud shadow"),
    palette=[[0, 0, 0], [255, 255, 255], [170, 170, 170], [85, 85, 85]],
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

source_train = dict(
    type="CLOUDSEN12HIGHL1CDataset",
    data_root=source_root,
    data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
    pipeline=train_pipeline,
)
target_selected = dict(
    type="Phase45L8ManifestDataset",
    manifest_path=manifest_path,
    split="target_train",
    selection_path=selection_path,
    target_to_source=True,
    metainfo=source_metainfo,
    pipeline=train_pipeline,
)

# 65 selected target patches x 131 repeats = 8,515 samples, matched to the
# 8,490-sample source replay set without changing either annotation budget.
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=[source_train, dict(type="RepeatDataset", times=131, dataset=target_selected)],
    ),
)
val_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="Phase45L8ManifestDataset",
        manifest_path=manifest_path,
        split="target_val",
        target_to_source=True,
        metainfo=source_metainfo,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
test_evaluator = val_evaluator

load_from = "work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
optim_wrapper = dict(optimizer=dict(lr=2e-5))
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=100),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=100, end=4000, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=4000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=1000,
        max_keep_ckpts=1, save_best="mIoU", rule="greater",
    )
)
randomness = dict(seed=50, deterministic=True)
work_dir = "./work_dirs/phase50_active_1pct/seed50"
