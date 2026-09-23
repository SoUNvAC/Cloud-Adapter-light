_base_ = ["./phase52_sparse_msre_1pct_l1c.py"]

# Single-variable follow-up to Phase 52A. Architecture, 16-token budget,
# injection positions, selected 65 patches, optimizer, seed and 4,000-iter
# schedule are unchanged. Only a parameter-free balanced shadow BCE is added.
model = dict(
    type="ShadowAuxFrozenHeadEncoderDecoder",
    shadow_aux_weight=0.5,
    shadow_index=3,
)

work_dir = "./work_dirs/phase53_shadow_aux_msre_1pct/seed52"
