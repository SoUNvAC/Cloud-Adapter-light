"""Minimal Phase 64B target screen: six-band source-only or six-band MsRE."""

_base_ = ["./phase64b_source_parent_six.py"]

import os


target = os.environ.get("PHASE64B_TARGET", "").strip().lower()
method = os.environ.get("PHASE64B_METHOD", "").strip().lower()
if target not in {"l8", "sparcs"}:
    raise ValueError("PHASE64B_TARGET must be l8 or sparcs")
if method not in {"source_only", "msre"}:
    raise ValueError("Phase64B-2 permits only source_only or msre")

parent_metainfo = dict(
    classes=("surface_visible", "cloud", "shadow"),
    palette=[[80, 130, 70], [245, 245, 245], [70, 70, 70]],
)
train_pipeline = [
    dict(type="LoadPhase64BSixBandImage"),
    dict(type="LoadAnnotations"),
    dict(type="FinalizePhase64BSpatialProtocol"),
    dict(type="RandomCrop", crop_size=(512, 512)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadPhase64BSixBandImage"),
    dict(type="LoadAnnotations"),
    dict(type="FinalizePhase64BSpatialProtocol"),
    dict(type="PackSegInputs"),
]

if target == "l8":
    manifest_path = os.environ.get(
        "PHASE64_L8_MANIFEST",
        "work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv",
    )
    shadow_metadata = os.environ.get(
        "PHASE64_L8_SHADOW_METADATA",
        "research_plans/protocol_data/l8_biome_usgs_shadow_status.csv",
    )
    raw_root = os.environ.get(
        "PHASE64B_L8_RAW_ROOT", "/home/scv/shared/data/l8_biome_raw"
    )
    common = dict(
        type="Phase64BL8SixBandParentDataset",
        manifest_path=manifest_path,
        raw_root=raw_root,
        usgs_shadow_metadata_path=shadow_metadata,
        usgs_shadows_filter="yes",
        metainfo=parent_metainfo,
    )
    train_dataset = dict(**common, split="target_train", pipeline=train_pipeline)
    val_dataset = dict(**common, split="target_val", pipeline=test_pipeline)
else:
    manifest_path = os.environ.get(
        "PHASE64_SPARCS_MANIFEST",
        "work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv",
    )
    common = dict(
        type="Phase64BSparcsSixBandParentDataset",
        manifest_path=manifest_path,
        granularity="parent",
        metainfo=parent_metainfo,
    )
    train_dataset = dict(**common, split="target_train", pipeline=train_pipeline)
    val_dataset = dict(**common, split="target_val", pipeline=test_pipeline)

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_dataset,
)
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)
test_dataloader = val_dataloader

if method == "source_only":
    model = dict(
        type="FrozenBackboneEncoderDecoder",
        test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(341, 341))
        if target == "sparcs"
        else dict(mode="whole"),
    )
else:
    model = dict(
        type="FrozenHeadEncoderDecoder",
        test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(341, 341))
        if target == "sparcs"
        else dict(mode="whole"),
        backbone=dict(
            type="TargetMsRECloudAdapterDinoVisionTransformer",
            in_chans=6,
            init_cfg=None,
            cloud_adapter_config=dict(in_channels=6),
            target_enabled=True,
            target_msre_config=dict(
                injection_indices=[2, 5, 8, 11],
                token_length=16,
                scale_init=1e-3,
            ),
        ),
    )

load_from = os.environ.get("PHASE64B_SOURCE_CHECKPOINT", "")
if not load_from:
    raise ValueError("PHASE64B_SOURCE_CHECKPOINT is required")
randomness = dict(seed=64, deterministic=True)
work_dir = f"./work_dirs/phase64b_target_parent/{target}/{method}/seed64"
