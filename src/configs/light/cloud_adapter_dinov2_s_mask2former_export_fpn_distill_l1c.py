_base_ = ["./cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"]

# Phase 13: use V8's deformable pixel decoder as a training-only feature
# teacher for the exportable Phase 12 FPN. The launcher supplies both the
# Phase 12 student initialization and the V8 teacher checkpoint.
model = dict(
    type="PixelFeatureDistillEncoderDecoder",
    teacher_config=(
        "configs/light/"
        "cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py"
    ),
    teacher_checkpoint=(
        "work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c/"
        "best_mIoU_iter_*.pth"
    ),
    teacher_fp16=True,
    feature_distill_weight=0.5,
    mask_feature_weight=1.0,
    pyramid_feature_weight=1.0,
)

# The teacher is training-only. Batch one keeps teacher feature extraction and
# the student's FP32 Hungarian matching within a 24-GiB 4090D.
train_dataloader = dict(batch_size=1)
optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    paramwise_cfg=dict(
        custom_keys={
            "decode_head.pixel_decoder": dict(lr_mult=5.0),
        },
    ),
)
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

work_dir = (
    "./work_dirs/"
    "cloud_adapter_dinov2_s_mask2former_export_fpn_distill_l1c"
)
