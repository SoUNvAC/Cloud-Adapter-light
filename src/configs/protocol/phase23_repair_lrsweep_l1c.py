_base_ = ["./phase23_clean_v12_l1c.py"]

import os

fpn_lr_mult = float(os.environ.get("PHASE23_FPN_LR_MULT", "1"))
if fpn_lr_mult not in {1.0, 2.0, 3.0, 4.0}:
    raise ValueError("Phase 23 repair sweep multiplier must be one of 1, 2, 3, 4")

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={"decode_head.pixel_decoder": dict(lr_mult=fpn_lr_mult)}
    )
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=500, end=8000, by_epoch=False),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=8000, val_interval=1000)

work_dir = "./work_dirs/phase23_repair_lrsweep"
