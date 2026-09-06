_base_ = ["./cloud_adapter_dinov2_s_light_v3_fpn_l1c.py"]

num_classes = 4

# Optional second V3 ablation. Deep supervision of the finest DINO feature is
# active only in loss mode; EncoderDecoder does not execute this head during
# prediction, so deployment latency is identical to plain V3-FPN.
model = dict(
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=384,
        in_index=0,
        channels=64,
        num_convs=1,
        kernel_size=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        align_corners=False,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=0.4,
                loss_name="loss_aux_ce",
            ),
            dict(
                type="DiceLoss",
                naive_dice=True,
                loss_weight=0.4,
                loss_name="loss_aux_dice",
            ),
        ],
    ),
)

work_dir = "./work_dirs/cloud_adapter_dinov2_s_light_v3_fpn_aux_l1c"
