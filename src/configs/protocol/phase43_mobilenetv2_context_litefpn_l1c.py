_base_ = ["./phase37_mobilenetv2_litefpn_weakclass_l1c.py"]

model = dict(
    decode_head=dict(
        pixel_decoder=dict(type="ContextLiteFPNPixelDecoder"),
    ),
)

work_dir = "./work_dirs/phase43_context_litefpn/seed42"
