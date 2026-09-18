_base_ = ["./phase23_clean_v12_l1c.py"]

model = dict(
    type="WeakClassBoundaryDistillEncoderDecoder",
    teacher_config="configs/protocol/phase22_clean_v8_l1c.py",
    teacher_checkpoint="__PHASE25_TEACHER_CHECKPOINT_MUST_BE_OVERRIDDEN__",
    distill_temperature=2.0,
    distill_weight=1.0,
    distill_class_weights=[1.0, 1.0, 2.0, 2.0],
    boundary_weight=2.0,
    boundary_radius=1,
    teacher_fp16=True,
)

# The frozen online V8 teacher is training-only. Batch one keeps FP32 Hungarian
# matching and the teacher inside the 24-GiB 4090D memory budget.
train_dataloader = dict(batch_size=1)

optim_wrapper = dict(optimizer=dict(lr=1e-5))
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=200, end=10000, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=10000, val_interval=1000)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        save_best="mIoU",
        rule="greater",
    ),
)

work_dir = "./work_dirs/phase25_block_kd"
