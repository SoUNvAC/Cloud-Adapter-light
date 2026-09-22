_base_ = ["./phase37_mobilenetv2_litefpn_weakclass_l1c.py"]

manifest_path = "work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv"
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
model = dict(
    decode_head=dict(
        # Target order is clear, shadow, thin, thick.  Preserve Phase 37's
        # semantic weak-class weights after reordering from the source labels.
        loss_cls=dict(class_weight=[1.0, 1.25, 1.5, 1.0, 0.1]),
    ),
)
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
work_dir = "./work_dirs/phase45_oracle_compact/seed42"
