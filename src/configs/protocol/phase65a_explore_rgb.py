"""User-authorized fit/development exploration; no confirmation dataloader."""
_base_ = ['./phase64_source_parent_rgb.py']
import os
custom_imports = dict(imports=['cloud_adapter.datasets.phase65_catalogue'], allow_failed_imports=False)
prepared = os.environ['PHASE65_PREPARED_MANIFEST']
source = os.environ['PHASE65_SOURCE_CHECKPOINT']
train_pipeline = [dict(type='LoadPhase65RGB'), dict(type='LoadPhase65Mask'),
                  dict(type='RandomCrop', crop_size=(512,512)), dict(type='RandomFlip', prob=0.5),
                  dict(type='PhotoMetricDistortion'), dict(type='PackSegInputs')]
test_pipeline = [dict(type='LoadPhase65RGB'), dict(type='Resize', scale=(512,512)),
                 dict(type='LoadPhase65Mask'), dict(type='PackSegInputs')]
train_dataloader = dict(_delete_=True, batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True), dataset=dict(type='Phase65CatalogueDataset',
    prepared_manifest=prepared, cohort='fit', pipeline=train_pipeline))
val_dataloader = dict(_delete_=True, batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), dataset=dict(type='Phase65CatalogueDataset',
    prepared_manifest=prepared, cohort='development_val', pipeline=test_pipeline))
test_dataloader = val_dataloader
model = dict(type='FrozenHeadEncoderDecoder', backbone=dict(
    type='TargetMsRECloudAdapterDinoVisionTransformer', init_cfg=None, target_enabled=True,
    target_msre_config=dict(injection_indices=[2,5,8,11], token_length=16, scale_init=1e-3)))
load_from = source
randomness = dict(seed=65, deterministic=True)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4000, val_interval=500)
work_dir = './work_dirs/phase65a_explore_20261008/msre_seed65'
