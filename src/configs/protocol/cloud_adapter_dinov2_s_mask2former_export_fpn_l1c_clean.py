_base_ = [
    "../light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py",
]

# Phase 21 protocol repair. Historical Phase 1--20 configurations deliberately
# remain unchanged for reproducibility. New experiments select checkpoints on
# the official validation partition and evaluate the test partition only after
# all model and hyperparameter choices have been frozen.
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

work_dir = "./work_dirs/phase21_v12_clean_protocol"
