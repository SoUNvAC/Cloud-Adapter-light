_base_ = ["../control/cloud_adapter_dinov2_s_mask2former_4x_l1c.py"]

# Structured compression of the 72.18-mIoU control decoder. Backbone,
# Cloud-Adapter, losses, data, and schedule stay unchanged.
model = dict(
    decode_head=dict(
        feat_channels=128,
        out_channels=128,
        num_queries=50,
        pixel_decoder=dict(
            encoder=dict(
                num_layers=3,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=128,
                        num_heads=4,
                    ),
                    ffn_cfg=dict(
                        embed_dims=128,
                        feedforward_channels=512,
                    ),
                ),
            ),
            positional_encoding=dict(num_feats=64),
        ),
        positional_encoding=dict(num_feats=64),
        transformer_decoder=dict(
            num_layers=3,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=128,
                    num_heads=4,
                ),
                cross_attn_cfg=dict(
                    embed_dims=128,
                    num_heads=4,
                ),
                ffn_cfg=dict(
                    embed_dims=128,
                    feedforward_channels=512,
                ),
            ),
        ),
    ),
)

# The compressed decoder should fit batch 2 in FP32 on a 24-GiB 4090D.
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = val_dataloader

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_lite_l1c"
