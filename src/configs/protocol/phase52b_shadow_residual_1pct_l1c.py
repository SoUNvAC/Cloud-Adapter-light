_base_ = ["./phase52_sparse_msre_1pct_l1c.py"]

# Freeze the selected Phase 52A model and train only a zero-initialized
# shadow/non-shadow residual. The original non-shadow class ratios are fixed.
model = dict(
    type="FrozenPhase52ShadowResidualEncoderDecoder",
    shadow_feature_index=2,
    shadow_index=3,
    shadow_hidden_channels=32,
    shadow_dilation=3,
    shadow_bce_weight=1.0,
    shadow_tversky_weight=1.0,
    shadow_boundary_weight=0.5,
    residual_enabled=True,
)

load_from = "work_dirs/phase52_sparse_msre_1pct/seed52/best_mIoU_iter_1000.pth"
train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type="ShadowGuaranteedInfiniteSampler",
        diagnosis_path="work_dirs/phase52_shadow_diagnosis/summary.json",
        batch_size=4,
        seed=52,
    ),
)
work_dir = "./work_dirs/phase52b_shadow_residual_1pct/seed52"
