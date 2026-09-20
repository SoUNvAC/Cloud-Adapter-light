_base_ = ["./phase23_clean_v12_l1c.py"]

import os

fpn_lr_mult = float(os.environ["PHASE23_FPN_LR_MULT"])
if fpn_lr_mult not in {1.0, 2.0, 3.0, 4.0}:
    raise ValueError("Selected Phase 23 repair multiplier is outside the preregistered sweep")

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={"decode_head.pixel_decoder": dict(lr_mult=fpn_lr_mult)}
    )
)

work_dir = "./work_dirs/phase23_repair_selected_seed42"
