_base_ = ["./cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c.py"]

# Single-variable V8 ablation: retain V7 width, queries, and query decoder,
# while reducing the multi-scale deformable pixel encoder from three layers
# to two. This targets spatial attention compute directly.
model = dict(
    decode_head=dict(pixel_decoder=dict(encoder=dict(num_layers=2))),
)

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c"
