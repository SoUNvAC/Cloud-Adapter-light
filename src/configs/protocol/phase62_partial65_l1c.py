_base_ = ["./phase62_common_partial_label_l1c.py"]

# Group 3: all 65 selected patches; raw ID0 in the 53 Shadows?=no patches is
# supervised by S_i={clear, cloud shadow}, while thin/thick remain ordinary.
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            usgs_shadows_filter=None,
            mask_unlabelled_shadow_clear=False,
            partial_unlabelled_shadow_clear=True,
        )
    )
)

work_dir = "./work_dirs/phase62_partial_labels/partial65/seed62"

