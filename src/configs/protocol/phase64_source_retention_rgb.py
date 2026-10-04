"""Evaluate a target-adapted Phase 64 model on CloudSEN source validation."""

_base_ = ["./phase64_target_parent_rgb.py"]

import os


method = os.environ.get("PHASE64_METHOD", "").strip().lower()
source_root = os.environ.get(
    "PHASE64_CLOUDSEN_ROOT", "data/cloudsen12_high_l1c"
)
parent_metainfo = dict(
    classes=("surface_visible", "cloud", "shadow"),
    palette=[[80, 130, 70], [245, 245, 245], [70, 70, 70]],
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
source_val_dataset = dict(
    type="Phase64CloudSENParentDataset",
    data_root=source_root,
    data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
    metainfo=parent_metainfo,
    pipeline=test_pipeline,
)
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=source_val_dataset,
)
val_dataloader = test_dataloader

# MsRE owns an explicit target residual path.  Source retention is measured
# with that path gated off, matching the preregistered residual-router policy.
if method == "msre":
    model = dict(backbone=dict(target_enabled=False))

work_dir = (
    f"./work_dirs/phase64_source_retention/"
    f"{os.environ.get('PHASE64_TARGET', 'unset')}/{method}/seed64"
)
