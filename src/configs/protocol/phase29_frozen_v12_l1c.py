_base_ = ["./phase29_common_v12_l1c.py"]

model = dict(
    type="FrozenBackboneEncoderDecoder",
    backbone=dict(
        _delete_=True,
        type="DinoVisionTransformer",
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        img_size=512,
        ffn_layer="mlp",
        init_values=1e-5,
        block_chunks=0,
        out_indices=[2, 5, 8, 11],
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2_s_converted_512x512.pth",
        ),
    ),
)

work_dir = "./work_dirs/phase29_fair_baselines/frozen/seed42"
