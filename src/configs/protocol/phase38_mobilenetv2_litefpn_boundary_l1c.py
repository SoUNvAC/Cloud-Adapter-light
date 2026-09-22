_base_ = ["./phase31_mobilenetv2_litefpn_l1c.py"]

model = dict(
    type="BoundarySupervisedEncoderDecoder",
    auxiliary_class_weights=[1.0, 1.0, 1.5, 1.25],
    auxiliary_boundary_weight=1.5,
    auxiliary_boundary_radius=1,
    auxiliary_loss_weight=0.5,
)

work_dir = "./work_dirs/phase38_boundary_supervision/seed42"
