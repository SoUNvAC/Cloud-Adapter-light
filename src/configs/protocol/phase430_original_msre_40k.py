_base_ = ['./phase64_source_parent_rgb.py']

custom_imports = dict(imports=['phase430_model'], allow_failed_imports=False)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='Phase430Backbone'),
)
# Fresh task modules. Only the inherited DINOv2 pretrained init_cfg is loaded.
load_from = None
resume = False
train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
test_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=500, end=40000, by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
default_hooks = dict(checkpoint=dict(interval=2000, max_keep_ckpts=3, save_best='mIoU', rule='greater'))
custom_hooks = [dict(type='Phase430AuditHook')]
randomness = dict(seed=42901, deterministic=True)
work_dir = '/home/scv/shared/phase430_20261102/work_dirs/original_msre'

