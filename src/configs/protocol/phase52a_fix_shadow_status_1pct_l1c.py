_base_ = ["./phase52_sparse_msre_1pct_l1c.py"]

usgs_shadow_metadata_path = (
    "research_plans/protocol_data/l8_biome_usgs_shadow_status.csv"
)

# Keep the exact frozen 65-patch selection. In Shadows?=no scenes, target ID 0
# is not a verified shadow negative, so it becomes ignore while thin/thick cloud
# remain supervised. This changes label validity only, not data selection,
# architecture, trainable parameters, optimizer, augmentations, or iterations.
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            usgs_shadow_metadata_path=usgs_shadow_metadata_path,
            mask_unlabelled_shadow_clear=True,
        )
    )
)

# Model selection and target reporting use only the USGS scenes for which a
# cloud-shadow truth mask exists. The sealed target-test split is not read.
val_dataloader = dict(
    dataset=dict(
        usgs_shadow_metadata_path=usgs_shadow_metadata_path,
        usgs_shadows_filter="yes",
    )
)
test_dataloader = val_dataloader

work_dir = "./work_dirs/phase52a_fix_shadow_status_1pct/seed52"
