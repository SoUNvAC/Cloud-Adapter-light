_base_ = [
    "../_base_/datasets/cloudsen12_high_l1c.py",
    "../_base_/default_runtime.py",
    "../_base_/models/cloud_adapter_dinov2_small.py",
]

num_classes = 4

model = dict(
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=(512, 512),
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
    backbone=dict(
        img_size=512,
        # Four evenly spaced interactions for the 12-block DINOv2-S.
        adapter_index=[2, 5, 8, 11],
        out_indices=[2, 5, 8, 11],
        cloud_adapter_config=dict(
            _delete_=True,
            type="CloudAdapter",
            cnn_type="pmaa",
            int_type="convnext",
            emd_dim=384,
            num_layers=4,
            context_dim=64,
            hidden_channels=64,
            depth=4,
            return_multi_feats=False,
            return_last_feature=False,
            local_groups=1,
            global_groups=1,
            rank_dim=8,
        ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2_s_converted_512x512.pth",
        ),
    ),
    decode_head=dict(
        _delete_=True,
        type="LightCloudHead",
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        channels=64,
        decoder_channels=64,
        num_fuse_blocks=2,
        dropout_ratio=0.1,
        num_classes=num_classes,
        align_corners=False,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
                loss_name="loss_ce",
            ),
            dict(
                type="DiceLoss",
                naive_dice=True,
                loss_weight=1.0,
                loss_name="loss_dice",
            ),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# The backbone's train() implementation freezes DINOv2 and leaves only the
# Cloud-Adapter trainable. The lightweight head remains trainable as usual.
optim_wrapper = dict(
    type="AmpOptimWrapper",
    constructor="PEFTOptimWrapperConstructor",
    loss_scale="dynamic",
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
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True)
test_dataloader = val_dataloader

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best="mIoU",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

randomness = dict(seed=42, deterministic=False)

work_dir = "./work_dirs/cloud_adapter_dinov2_s_light_l1c"
