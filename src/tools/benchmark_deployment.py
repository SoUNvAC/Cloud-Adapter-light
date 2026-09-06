import argparse
import os
from pathlib import Path
import statistics
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end segmentation inference"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def main():
    args = parse_args()

    if args.batch_size <= 0 or args.input_size <= 0:
        raise ValueError("batch-size and input-size must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmseg.registry import MODELS
    from mmseg.structures import SegDataSample

    import cloud_adapter.models  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model = model.cuda().eval()

    size = args.input_size
    inputs = torch.randn(args.batch_size, 3, size, size, device="cuda")
    data_samples = [
        SegDataSample(
            metainfo=dict(
                ori_shape=(size, size),
                img_shape=(size, size),
                pad_shape=(size, size),
                padding_size=[0, 0, 0, 0],
                flip=False,
            )
        )
        for _ in range(args.batch_size)
    ]
    use_amp = args.precision == "fp16"

    torch.backends.cudnn.benchmark = True

    def infer_once():
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            return model(inputs, data_samples=data_samples, mode="predict")

    for _ in range(args.warmup):
        infer_once()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    latencies_ms = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = infer_once()
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    del output

    total_params = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_size = Path(args.checkpoint).stat().st_size / 2**20
    mean_ms = statistics.fmean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    p90_ms = percentile(latencies_ms, 0.90)
    throughput = 1000.0 * args.batch_size / mean_ms

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Precision: {args.precision}")
    print(f"Input: {args.batch_size} x 3 x {size} x {size}")
    print(f"Parameters: {total_params / 1e6:.3f} M")
    print(f"Checkpoint size: {checkpoint_size:.2f} MiB")
    print(f"Latency mean: {mean_ms:.3f} ms/batch")
    print(f"Latency median: {median_ms:.3f} ms/batch")
    print(f"Latency p90: {p90_ms:.3f} ms/batch")
    print(f"Throughput: {throughput:.2f} images/s")
    print(
        "Peak allocated memory: "
        f"{torch.cuda.max_memory_allocated() / 2**30:.3f} GiB"
    )


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
