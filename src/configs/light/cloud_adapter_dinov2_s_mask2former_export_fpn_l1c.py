_base_ = ["./cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py"]

# Phase 12: retain V8's compact query decoder but replace the deformable pixel
# encoder with a standard-op depthwise-separable FPN. The shell launcher loads
# V8 first, so the backbone, adapter, query embeddings, and transformer decoder
# are warm-started while this pixel decoder is initialized from scratch.
model = dict(
    decode_head=dict(
        pixel_decoder=dict(
            _delete_=True,
            type="LiteFPNPixelDecoder",
            strides=[4, 8, 16, 32],
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            # Mask2FormerHead checks this value before building the decoder.
            # LiteFPNPixelDecoder itself does not construct this encoder.
            encoder=dict(
                layer_cfg=dict(self_attn_cfg=dict(num_levels=3)),
            ),
        ),
    ),
)

# Protect the already-trained V8 components with a low base LR while allowing
# the new FPN pixel decoder to learn five times faster.
optim_wrapper = dict(
    optimizer=dict(lr=2e-5),
    paramwise_cfg=dict(
        custom_keys={
            "decode_head.pixel_decoder": dict(lr_mult=5.0),
        },
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=500, end=20000, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=1000)
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

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
