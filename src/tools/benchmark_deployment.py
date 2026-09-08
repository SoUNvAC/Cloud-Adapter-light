import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
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
    parser.add_argument(
        "--output-format",
        choices=("text", "tsv", "json"),
        default="text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress model initialization and checkpoint loading logs",
    )
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


@contextmanager
def suppress_output(enabled):
    if not enabled:
        yield
        return
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


def main():
    args = parse_args()

    if args.batch_size <= 0 or args.input_size <= 0:
        raise ValueError("batch-size and input-size must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")

    quiet = args.quiet or args.output_format != "text"
    with suppress_output(quiet):
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

    with suppress_output(quiet):
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
    result = dict(
        gpu=torch.cuda.get_device_name(0),
        precision=args.precision,
        batch_size=args.batch_size,
        input_size=size,
        parameters_m=total_params / 1e6,
        checkpoint_mib=checkpoint_size,
        latency_mean_ms=mean_ms,
        latency_median_ms=median_ms,
        latency_p90_ms=p90_ms,
        throughput_images_s=throughput,
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
    )

    if args.output_format == "json":
        print(json.dumps(result, ensure_ascii=False))
    elif args.output_format == "tsv":
        print(
            "\t".join(
                [
                    result["gpu"],
                    f'{result["parameters_m"]:.3f}',
                    f'{result["checkpoint_mib"]:.2f}',
                    f'{result["latency_mean_ms"]:.3f}',
                    f'{result["latency_median_ms"]:.3f}',
                    f'{result["latency_p90_ms"]:.3f}',
                    f'{result["throughput_images_s"]:.2f}',
                    f'{result["peak_memory_gib"]:.3f}',
                ]
            )
        )
    else:
        print(f'GPU: {result["gpu"]}')
        print(f'Precision: {result["precision"]}')
        print(f'Input: {args.batch_size} x 3 x {size} x {size}')
        print(f'Parameters: {result["parameters_m"]:.3f} M')
        print(f'Checkpoint size: {result["checkpoint_mib"]:.2f} MiB')
        print(f'Latency mean: {result["latency_mean_ms"]:.3f} ms/batch')
        print(f'Latency median: {result["latency_median_ms"]:.3f} ms/batch')
        print(f'Latency p90: {result["latency_p90_ms"]:.3f} ms/batch')
        print(f'Throughput: {result["throughput_images_s"]:.2f} images/s')
        print(f'Peak allocated memory: {result["peak_memory_gib"]:.3f} GiB')


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
