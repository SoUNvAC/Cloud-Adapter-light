_base_ = ["../light/cloud_adapter_dinov2_s_light_l1c.py"]

num_classes = 4

# Diagnostic control: the backbone and four Cloud-Adapter interactions are
# identical to the light student, while the decoder is restored to the
# original Mask2Former design. The score gap to V1/V2 measures decoder loss.
model = dict(
    backbone=dict(has_cat=False),
    decode_head=dict(
        _delete_=True,
        type="Mask2FormerHead",
        in_channels=[384, 384, 384, 384],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type="mmdet.DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost", weight=2.0),
                    dict(
                        type="mmdet.CrossEntropyLossCost",
                        weight=5.0,
                        use_sigmoid=True,
                    ),
                    dict(
                        type="mmdet.DiceCost",
                        weight=5.0,
                        pred_act=True,
                        eps=1.0,
                    ),
                ],
            ),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        ),
    ),
)

# Mask2Former is only a diagnostic upper-bound decoder and uses more memory.
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = val_dataloader

work_dir = "./work_dirs/cloud_adapter_dinov2_s_mask2former_4x_l1c"
