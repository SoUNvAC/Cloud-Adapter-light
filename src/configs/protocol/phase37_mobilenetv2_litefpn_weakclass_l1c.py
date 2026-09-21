_base_ = ["./phase31_mobilenetv2_litefpn_l1c.py"]

model = dict(
    decode_head=dict(
        loss_cls=dict(
            class_weight=[1.0, 1.0, 1.5, 1.25, 0.1],
        ),
    ),
)

work_dir = "./work_dirs/phase37_weakclass_supervision/seed42"
