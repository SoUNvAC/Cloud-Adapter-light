_base_ = ["./cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py"]

# V9 width compression: retain V8 queries and layer counts while halving the
# Mask2Former feature width. Dependent attention, FFN, and positional widths
# scale together to keep every transformer interface dimensionally valid.
model = dict(
    decode_head=dict(
        feat_channels=64,
        out_channels=64,
        pixel_decoder=dict(
            encoder=dict(
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=64,
                        num_heads=4,
                    ),
                    ffn_cfg=dict(
                        embed_dims=64,
                        feedforward_channels=256,
                    ),
                ),
            ),
            positional_encoding=dict(num_feats=32),
        ),
        positional_encoding=dict(num_feats=32),
        transformer_decoder=dict(
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=64,
                    num_heads=4,
                ),
                cross_attn_cfg=dict(
                    embed_dims=64,
                    num_heads=4,
                ),
                ffn_cfg=dict(
                    embed_dims=64,
                    feedforward_channels=256,
                ),
            ),
        ),
    ),
)

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_micro_q25_d2_p2_l1c"
