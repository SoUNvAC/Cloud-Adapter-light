_base_ = ["./phase22_clean_v8_l1c.py"]

model = dict(
    type="FactorizedEncoderDecoder",
    factor_loss_weight=1.0,
    reconstruction_loss_weight=1.0,
    freeze_base=True,
)

load_from = "work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
train_cfg = dict(type="IterBasedTrainLoop", max_iters=4000, val_interval=1000)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=100),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=100, end=4000, by_epoch=False),
]
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=1000, max_keep_ckpts=1)
)
randomness = dict(seed=42, deterministic=True)
work_dir = "./work_dirs/phase46_factorized_v8/seed42"
