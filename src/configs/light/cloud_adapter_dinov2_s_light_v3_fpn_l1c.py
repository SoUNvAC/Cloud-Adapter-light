_base_ = ["./cloud_adapter_dinov2_s_light_l1c.py"]

num_classes = 4

# V3 keeps the V1 backbone exactly: frozen DINOv2-S plus four Cloud-Adapter
# interactions. Only the deployment decoder changes, making this a clean
# decoder ablation against V1.
model = dict(
    backbone=dict(has_cat=False),
    decode_head=dict(
        _delete_=True,
        type="LightCloudFPNHead",
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        channels=64,
        decoder_channels=64,
        num_refine_blocks=1,
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
)

work_dir = "./work_dirs/cloud_adapter_dinov2_s_light_v3_fpn_l1c"
