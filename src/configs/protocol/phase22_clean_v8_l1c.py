_base_ = [
    "../light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py",
]

import os

# Accept either the historical src/data layout or a repository-level data
# directory selected by the one-click runner, without creating a symlink.
phase22_data_root = os.environ.get(
    "CLOUD_ADAPTER_DATA_ROOT", "data/cloudsen12_high_l1c"
)
train_dataloader = dict(dataset=dict(data_root=phase22_data_root))

# Phase 22 retrains V8 from the common DINOv2-S initialization. Checkpoint
# selection is allowed to observe only the official validation partition.
val_dataloader = dict(
    dataset=dict(
        data_root=phase22_data_root,
        data_prefix=dict(
            img_path="img_dir/val",
            seg_map_path="ann_dir/val",
        ),
    ),
)
test_dataloader = dict(
    dataset=dict(
        data_root=phase22_data_root,
        data_prefix=dict(
            img_path="img_dir/test",
            seg_map_path="ann_dir/test",
        ),
    ),
)

randomness = dict(seed=42, deterministic=True)
# PyTorch 2.1 has no deterministic CUDA implementation for the cumsum used by
# Mask2Former sine positional encoding. Keep deterministic algorithms enabled
# everywhere else, but warn instead of aborting for that documented exception.
deterministic_warn_only = True

work_dir = "./work_dirs/phase22_clean_v8_seed42"
