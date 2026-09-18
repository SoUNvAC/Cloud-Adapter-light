_base_ = ["./phase29_common_v12_l1c.py"]

# Direct 40k training: unlike Phase 23, this fair baseline does not warm-start
# the V12 head or Cloud-Adapter from a Phase 22 V8 checkpoint.
work_dir = "./work_dirs/phase29_fair_baselines/cloud_adapter/seed42"
