"""Phase 64B source model: the Phase 64 parent task with six-band input only."""

_base_ = ["./phase64_source_parent_rgb.py"]

import json
import os


stats_path = os.environ.get(
    "PHASE64B_PROTOCOL_SUMMARY",
    "work_dirs/phase64b_input_protocol/summary.json",
)
if not os.path.isfile(stats_path):
    raise FileNotFoundError(f"Run the frozen Phase64B-0 audit first: {stats_path}")
protocol = json.loads(open(stats_path, encoding="utf-8").read())
if protocol.get("target_test_reads") != 0:
    raise RuntimeError("Phase64B protocol must certify zero target-test reads")
stats = protocol["source_train_stats"]

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

train_pipeline = [
    dict(type="LoadPhase64BSixBandImage"),
    dict(type="LoadAnnotations"),
    dict(type="FinalizePhase64BSpatialProtocol"),
    dict(type="RandomCrop", crop_size=(256, 256)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadPhase64BSixBandImage"),
    dict(type="LoadAnnotations"),
    dict(type="FinalizePhase64BSpatialProtocol"),
    dict(type="PackSegInputs"),
]

model = dict(
    type="Phase64BSourceEncoderDecoder",
    data_preprocessor=dict(
        mean=stats["mean"],
        std=stats["std"],
        bgr_to_rgb=False,
        size=(256, 256),
    ),
    backbone=dict(
        in_chans=6,
        save_input_stem=True,
        init_cfg=None,
        cloud_adapter_config=dict(in_channels=6),
    ),
)

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="Phase64BCloudSENSixBandParentDataset",
        memmap_root=memmap_root,
        label_root=label_root,
        split="train",
        metainfo=parent_metainfo,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
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
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

load_from = os.environ.get(
    "PHASE64B_SOURCE_INIT",
    "work_dirs/phase64b_source_init/source_six_channel_init.pth",
)
randomness = dict(seed=64, deterministic=True)
work_dir = "./work_dirs/phase64b_source_parent/seed64"
