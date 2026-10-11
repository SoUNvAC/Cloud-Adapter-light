"""Phase80 fixed LoRA; identical Phase65a data/optimizer/scheduler."""
_base_ = ['./phase64_source_parent_rgb.py']
import os
custom_imports = dict(imports=['cloud_adapter.datasets.phase65_catalogue'], allow_failed_imports=False)
prepared = os.environ['PHASE65_PREPARED_MANIFEST']
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
    type='Phase64LoRACloudAdapterDinoVisionTransformer', init_cfg=None,
    lora_rank=4, lora_alpha=4., lora_dropout=0., lora_targets=('qkv','proj','fc1','fc2')))
load_from = None  # In-memory strict Source conversion in run_phase80_training.py.
randomness = dict(seed=65, deterministic=True)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4000, val_interval=500)
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/phase80_20261011/lora_seed65'
