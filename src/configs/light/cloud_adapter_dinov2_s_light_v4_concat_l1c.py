_base_ = ["./cloud_adapter_dinov2_s_light_l1c.py"]

# Clean fusion ablation against V1: the backbone, four adapters, 64-channel
# width, refinement blocks, losses, and schedule are unchanged. The only
# difference is preserving the four projected scales along the channel axis
# before a learned 1x1 fusion instead of averaging them.
model = dict(
    backbone=dict(has_cat=False),
    decode_head=dict(fusion="concat"),
)

work_dir = "./work_dirs/cloud_adapter_dinov2_s_light_v4_concat_l1c"
