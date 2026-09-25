_base_ = ["./phase52_sparse_msre_1pct_l1c.py"]

usgs_shadow_metadata_path = (
    "research_plans/protocol_data/l8_biome_usgs_shadow_status.csv"
)

# One parameter-free wrapper is shared by all three groups. It reproduces the
# native LightCloudHead CE/Dice path and adds set-valued NLL only when label 254
# is present, so groups 1/2 have an exactly zero partial term.
model = dict(
    type="PartialLabelFrozenHeadEncoderDecoder",
    partial_label_index=254,
    partial_classes=(0, 3),
    partial_loss_weight=1.0,
)

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            usgs_shadow_metadata_path=usgs_shadow_metadata_path,
            partial_label_index=254,
        )
    )
)

# Checkpoint selection and all reported target metrics use only scenes with a
# verified shadow mask. The target-test and CloudSEN internal-test stay sealed.
val_dataloader = dict(
    dataset=dict(
        usgs_shadow_metadata_path=usgs_shadow_metadata_path,
        usgs_shadows_filter="yes",
    )
)
test_dataloader = val_dataloader

randomness = dict(seed=62, deterministic=True)

