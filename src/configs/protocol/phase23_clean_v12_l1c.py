_base_ = [
    "../light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py",
]

# Phase 23 pairs every V12 run with the clean V8 checkpoint trained using the
# same seed in Phase 22. Only validation may influence checkpoint selection.
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path="img_dir/val",
            seg_map_path="ann_dir/val",
        ),
    ),
)
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path="img_dir/test",
            seg_map_path="ann_dir/test",
        ),
    ),
)

randomness = dict(seed=42, deterministic=True)
# Keep the documented Phase 22 PyTorch 2.1 exception consistent. Mask2Former
# positional encoding and CUDA IoU histograms do not have deterministic CUDA
# implementations, so aborting here would make the paired protocol differ from
# the clean V8 baseline solely because of framework support.
deterministic_warn_only = True

work_dir = "./work_dirs/phase23_clean_v12_seed42"
