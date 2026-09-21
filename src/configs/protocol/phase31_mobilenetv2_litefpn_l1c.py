_base_ = ["./phase29_mobilenetv2_v8_l1c.py"]

model = dict(
    decode_head=dict(
        pixel_decoder=dict(
            _delete_=True,
            type="LiteFPNPixelDecoder",
            strides=[4, 8, 16, 32],
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            encoder=dict(
                layer_cfg=dict(self_attn_cfg=dict(num_levels=3)),
            ),
        ),
    ),
)

work_dir = "./work_dirs/phase31_mobilenetv2_litefpn/seed42"
