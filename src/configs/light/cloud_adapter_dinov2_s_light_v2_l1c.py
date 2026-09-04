_base_ = ["./cloud_adapter_dinov2_s_light_l1c.py"]

# V2 keeps the same frozen DINOv2-S and cheap decoder as V1. It exposes the
# PMAA convolutional cache to every output scale, preserving local boundaries
# that may be weak in pure ViT features, and learns four scale-fusion weights.
model = dict(
    backbone=dict(has_cat=True),
    decode_head=dict(
        in_channels=[448, 448, 448, 448],
        fusion="weighted_sum",
    ),
)

work_dir = "./work_dirs/cloud_adapter_dinov2_s_light_v2_l1c"
