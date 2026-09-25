_base_ = ["./phase62_common_partial_label_l1c.py"]

# Group 1: only the 12 selected patches from Shadows?=yes scenes.
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            usgs_shadows_filter="yes",
            mask_unlabelled_shadow_clear=False,
            partial_unlabelled_shadow_clear=False,
        )
    )
)

work_dir = "./work_dirs/phase62_partial_labels/complete12/seed62"

