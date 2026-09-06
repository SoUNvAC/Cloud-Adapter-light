import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the light training setup")
    parser.add_argument(
        "--config",
        default="configs/light/cloud_adapter_dinov2_s_light_l1c.py",
    )
    parser.add_argument("--skip-forward", action="store_true")
    return parser.parse_args()


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def reduce_loss_value(value, torch):
    if torch.is_tensor(value):
        return value.mean()
    if isinstance(value, (list, tuple)):
        return sum(item.mean() for item in value)
    raise TypeError(f"Unsupported loss value type: {type(value)!r}")


def main():
    args = parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.structures import PixelData
    from mmseg.registry import MODELS
    from mmseg.structures import SegDataSample

    import cloud_adapter.models  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    checkpoint = Path(cfg.model.backbone.init_cfg.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing converted DINOv2-S checkpoint: {checkpoint}. "
            "Run: bash tools/prepare_dinov2_small.sh"
        )

    data_root = Path(cfg.train_dataloader.dataset.data_root)
    expected_dirs = [
        data_root / "img_dir" / "train",
        data_root / "ann_dir" / "train",
        data_root / "img_dir" / "test",
        data_root / "ann_dir" / "test",
    ]
    missing = [str(path) for path in expected_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))

    for path in expected_dirs:
        print(f"{path}: {count_files(path)} files")

    model = MODELS.build(cfg.model)
    model.init_weights()
    model.train()

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total parameters: {total / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable / 1e6:.2f} M")
    print(f"Trainable ratio: {100.0 * trainable / total:.2f}%")

    if not args.skip_forward:
        model = model.cuda()
        dummy = torch.randn(1, 3, 512, 512, device="cuda")
        data_sample = SegDataSample(
            metainfo=dict(
                ori_shape=(512, 512),
                img_shape=(512, 512),
                pad_shape=(512, 512),
                padding_size=[0, 0, 0, 0],
                flip=False,
            )
        )

        decode_head_type = str(cfg.model.decode_head.type)
        needs_train_smoke = (
            cfg.model.get("auxiliary_head") is not None
            or "Mask2Former" in decode_head_type
        )
        if needs_train_smoke:
            model.train()
            num_classes = cfg.model.decode_head.num_classes
            target = torch.randint(
                0, num_classes, (1, 512, 512), device="cuda", dtype=torch.long
            )
            data_sample.gt_sem_seg = PixelData(data=target)
            use_amp = cfg.optim_wrapper.type == "AmpOptimWrapper"
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                losses = model(dummy, data_samples=[data_sample], mode="loss")
                total_loss = sum(
                    reduce_loss_value(value, torch)
                    for name, value in losses.items()
                    if "loss" in name
                )
            total_loss.backward()
            print(f"Training smoke loss: {total_loss.detach().item():.4f}")
            model.zero_grad(set_to_none=True)
            del losses, total_loss, target

        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            # Mask2Former needs batch metadata even for inference, whereas
            # mode="tensor" calls its head without batch_data_samples.
            output = model(dummy, data_samples=[data_sample], mode="predict")
        prediction = output[0].pred_sem_seg.data
        print(f"Forward output shape: {tuple(prediction.shape)}")
        print(f"Peak allocated memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    print("Setup check passed")


if __name__ == "__main__":
    # tools/train.py changes to the repository root; do the same when invoked
    # from another directory.
    os.chdir(REPO_ROOT)
    main()
