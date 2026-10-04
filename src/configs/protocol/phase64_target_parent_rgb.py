"""Frozen single-seed Phase 64 target matrix.

Select exactly one target and one method before parsing this config:
PHASE64_TARGET={l8,sparcs}
PHASE64_METHOD={shared_parent,full,lora,msre}
"""

_base_ = ["./phase64_source_parent_rgb.py"]

import os


target = os.environ.get("PHASE64_TARGET", "").strip().lower()
method = os.environ.get("PHASE64_METHOD", "").strip().lower()
if target not in {"l8", "sparcs"}:
    raise ValueError("PHASE64_TARGET must be exactly 'l8' or 'sparcs'")
if method not in {"shared_parent", "full", "lora", "msre"}:
    raise ValueError(
        "PHASE64_METHOD must be shared_parent, full, lora, or msre"
    )

parent_metainfo = dict(
    classes=("surface_visible", "cloud", "shadow"),
    palette=[[80, 130, 70], [245, 245, 245], [70, 70, 70]],
)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=crop_size),
    dict(type="LoadAnnotations"),
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
    train_dataset = dict(
        type="Phase64L8ParentDataset",
        manifest_path=manifest_path,
        split="target_train",
        usgs_shadow_metadata_path=shadow_metadata,
        usgs_shadows_filter="yes",
        metainfo=parent_metainfo,
        pipeline=train_pipeline,
    )
    val_dataset = dict(
        type="Phase64L8ParentDataset",
        manifest_path=manifest_path,
        split="target_val",
        usgs_shadow_metadata_path=shadow_metadata,
        usgs_shadows_filter="yes",
        metainfo=parent_metainfo,
        pipeline=test_pipeline,
    )
else:
    sparcs_manifest = os.environ.get(
        "PHASE64_SPARCS_MANIFEST",
        "work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv",
    )
    train_dataset = dict(
        type="Phase64SparcsManifestDataset",
        manifest_path=sparcs_manifest,
        split="target_train",
        granularity="parent",
        metainfo=parent_metainfo,
        pipeline=train_pipeline,
    )
    val_dataset = dict(
        type="Phase64SparcsManifestDataset",
        manifest_path=sparcs_manifest,
        split="target_val",
        granularity="parent",
        metainfo=parent_metainfo,
        pipeline=test_pipeline,
    )

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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)
test_dataloader = val_dataloader

if method == "shared_parent":
    model = dict(type="FrozenBackboneEncoderDecoder")
elif method == "full":
    model = dict(
        type="EncoderDecoder",
        backbone=dict(type="Phase64FullFineTuneCloudAdapterDinoVisionTransformer"),
    )
elif method == "lora":
    model = dict(
        type="FrozenHeadEncoderDecoder",
        backbone=dict(
            type="Phase64LoRACloudAdapterDinoVisionTransformer",
            lora_rank=4,
            lora_alpha=4.0,
            lora_dropout=0.0,
            lora_targets=("qkv", "proj", "fc1", "fc2"),
        ),
    )
else:
    model = dict(
        type="FrozenHeadEncoderDecoder",
        backbone=dict(
            type="TargetMsRECloudAdapterDinoVisionTransformer",
            target_enabled=True,
            target_msre_config=dict(
                injection_indices=[2, 5, 8, 11],
                token_length=16,
                scale_init=1e-3,
            ),
        ),
    )

load_from = os.environ.get("PHASE64_PARENT_CHECKPOINT", "")
if not load_from:
    raise ValueError("PHASE64_PARENT_CHECKPOINT must identify the frozen source-parent model")

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
work_dir = f"./work_dirs/phase64_target_parent/{target}/{method}/seed64"
