_base_ = [
    "./cloud_adapter_dinov2_s_mask2former_export_fpn_l1c_clean.py",
]

# Fair PEFT comparison: every method starts from the same pretrained DINOv2-S
# and a randomly initialized, structurally identical V12 deployment head.
load_from = None

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "pos_embed": dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=500, end=40000, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=2000)

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
test_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best="mIoU",
        rule="greater",
    ),
)

randomness = dict(seed=42, deterministic=True)
work_dir = "./work_dirs/phase29_common_v12"
