_base_ = ["./phase22_clean_v8_l1c.py"]

source_checkpoint = "work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
pseudo_manifest = "work_dirs/phase45_teacher_student/pseudo_manifest.csv"
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

load_from = source_checkpoint
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="Phase45L8ManifestDataset",
        manifest_path=pseudo_manifest,
        split="target_train",
        mask_column="pseudo_mask_path",
        metainfo=source_metainfo,
        pipeline=train_pipeline,
    ),
)
train_cfg = dict(type="IterBasedTrainLoop", max_iters=10000, val_interval=10001)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=500, end=10000, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=10000, max_keep_ckpts=1)
)
randomness = dict(seed=42, deterministic=True)
work_dir = "./work_dirs/phase45_teacher_student/student"
