_base_ = ["./phase31_mobilenetv2_litefpn_l1c.py"]

model = dict(
    decode_head=dict(
        pixel_decoder=dict(
            type="BiLiteFPNPixelDecoder",
            fusion_epsilon=1e-4,
        ),
    ),
)

work_dir = "./work_dirs/phase35_bilitefpn/seed42"
