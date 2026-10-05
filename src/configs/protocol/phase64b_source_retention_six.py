"""Evaluate a six-band Phase64B MsRE checkpoint on CloudSEN source-val."""

_base_ = ["./phase64b_target_parent_six.py"]

import os


if os.environ.get("PHASE64B_METHOD", "").strip().lower() != "msre":
    raise ValueError("Phase64B source-retention config is defined only for MsRE")

model = dict(
    backbone=dict(target_enabled=False),
    test_cfg=dict(mode="whole"),
)
memmap_root = os.environ.get(
    "PHASE64B_CLOUDSEN_MEMMAP_ROOT",
    "../phase64b_data/cloudsen12_high_raw",
)
label_root = os.environ.get(
    "PHASE64_CLOUDSEN_ROOT",
    "data/cloudsen12_high_l1c",
)
parent_metainfo = dict(
    classes=("surface_visible", "cloud", "shadow"),
    palette=[[80, 130, 70], [245, 245, 245], [70, 70, 70]],
)
source_test_pipeline = [
    dict(type="LoadPhase64BSixBandImage"),
    dict(type="LoadAnnotations"),
    dict(type="FinalizePhase64BSpatialProtocol"),
    dict(type="PackSegInputs"),
]
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="Phase64BCloudSENSixBandParentDataset",
        memmap_root=memmap_root,
        label_root=label_root,
        split="val",
        metainfo=parent_metainfo,
        pipeline=source_test_pipeline,
    ),
)
val_dataloader = test_dataloader
work_dir = (
    "./work_dirs/phase64b_source_retention/"
    f"{os.environ.get('PHASE64B_TARGET', 'unset')}/msre/seed64"
)
