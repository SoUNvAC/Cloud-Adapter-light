_base_ = ["./phase46_factorized_v8_l1c.py"]

model = dict(
    type="PixelFeatureFactorizedEncoderDecoder",
    mask_feature_channels=128,
)
work_dir = "./work_dirs/phase47_pixel_factorized_v8/seed42"
