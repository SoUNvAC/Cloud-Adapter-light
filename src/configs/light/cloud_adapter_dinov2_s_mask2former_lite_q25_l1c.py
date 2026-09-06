_base_ = ["./cloud_adapter_dinov2_s_mask2former_lite_l1c.py"]

# Single-variable decoder compression: halve object queries while keeping the
# successful V5 width and layer counts unchanged.
model = dict(decode_head=dict(num_queries=25))

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_l1c"
