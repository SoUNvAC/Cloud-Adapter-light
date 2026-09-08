_base_ = ["./cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py"]

# Fine-tune the selected V8 Pareto student with semantic-logit knowledge from
# the 72.18-mIoU Control. The training script supplies the V8 student weights
# through load_from. Teacher weights are never saved into student checkpoints.
model = dict(
    type="LogitDistillEncoderDecoder",
    teacher_config="configs/control/cloud_adapter_dinov2_s_mask2former_4x_l1c.py",
    teacher_checkpoint=(
        "work_dirs/cloud_adapter_dinov2_s_mask2former_4x_l1c/"
        "best_mIoU_iter_*.pth"
    ),
    distill_temperature=2.0,
    distill_weight=2.0,
    teacher_fp16=True,
)

# Online teacher inference plus FP32 Hungarian student training is kept at
# batch one for the 24-GiB 4090D.
train_dataloader = dict(batch_size=1)

optim_wrapper = dict(optimizer=dict(lr=2e-5))
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

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_v8_distill_l1c"
