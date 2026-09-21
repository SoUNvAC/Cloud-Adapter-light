_base_ = ["./phase26_resnet18_v8_l1c.py"]

import os

l8_root = os.environ.get("PHASE28_L8_ROOT", "data/l8_biome")
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

model = dict(
    type="LabelRemapEncoderDecoder",
    label_remap=[0, 3, 2, 1],
)
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="L8BIOMEDataset",
        data_root=l8_root,
        data_prefix=dict(img_path="img_dir/test", seg_map_path="ann_dir/test"),
        pipeline=test_pipeline,
    ),
)
test_evaluator = dict(
    type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"]
)
