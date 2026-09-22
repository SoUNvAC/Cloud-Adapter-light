_base_ = ["./phase22_clean_v8_l1c.py"]

manifest_path = "work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv"
crop_size = (512, 512)
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
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="Phase45L8ManifestDataset",
        manifest_path=manifest_path,
        split="target_train",
        pipeline=train_pipeline,
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
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
test_evaluator = val_evaluator
randomness = dict(seed=42, deterministic=True)
work_dir = "./work_dirs/phase45_oracle_v8/seed42"
