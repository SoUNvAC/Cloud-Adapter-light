_base_ = ["./cloud_adapter_dinov2_s_mask2former_lite_q25_l1c.py"]

# Single-variable V7 ablation: retain V6 width, queries, and pixel encoder,
# while reducing the query Transformer decoder from three layers to two.
model = dict(decode_head=dict(transformer_decoder=dict(num_layers=2)))

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c"
