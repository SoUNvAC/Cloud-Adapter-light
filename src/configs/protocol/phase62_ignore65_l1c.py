_base_ = ["./phase62_common_partial_label_l1c.py"]

# Group 2: all 65 selected patches; raw ID0 in the 53 Shadows?=no patches is
# ignored while thin/thick pixels retain ordinary CE/Dice supervision.
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            usgs_shadows_filter=None,
            mask_unlabelled_shadow_clear=True,
            partial_unlabelled_shadow_clear=False,
        )
    )
)

work_dir = "./work_dirs/phase62_partial_labels/ignore65/seed62"

