_base_ = [
    "../light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py",
]

# Phase 22 retrains V8 from the common DINOv2-S initialization. Checkpoint
# selection is allowed to observe only the official validation partition.
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

work_dir = "./work_dirs/phase22_clean_v8_seed42"
