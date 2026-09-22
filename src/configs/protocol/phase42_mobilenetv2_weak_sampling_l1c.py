_base_ = ["./phase37_mobilenetv2_litefpn_weakclass_l1c.py"]

train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type="WeakPrevalenceInfiniteSampler",
        manifest_path="work_dirs/phase42_weak_sampling/sampling_manifest.json",
        rich_probability=0.5,
        seed=42,
    ),
)

work_dir = "./work_dirs/phase42_weak_sampling/seed42"
