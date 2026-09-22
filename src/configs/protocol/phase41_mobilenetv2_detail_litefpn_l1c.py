_base_ = ["./phase37_mobilenetv2_litefpn_weakclass_l1c.py"]

model = dict(
    decode_head=dict(
        pixel_decoder=dict(
            type="DetailLiteFPNPixelDecoder",
            detail_channels=16,
            detail_gate_init=0.1,
        ),
    ),
)

work_dir = "./work_dirs/phase41_detail_litefpn/seed42"
