_base_ = ["./phase22_clean_v8_l1c.py"]

import os

mlp_ratio = float(os.environ.get("PHASE25_MLP_RATIO", "4.0"))
if mlp_ratio not in {4.0, 3.0, 2.5, 2.0}:
    raise ValueError("Phase 25 MLP ratio is outside the preregistered screen")

model = dict(backbone=dict(mlp_ratio=mlp_ratio))
work_dir = "./work_dirs/phase25_mlp_pruned_v8"
