import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validate_phase12_onnxruntime import (  # noqa: E402
    list_images,
    load_rgb_image,
)
from validate_phase12_tensorrt import TensorRTRunner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 18 end-to-end deployment benchmark"
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument(
        "--image-dir", default="data/cloudsen12_high_l1c/img_dir/test"
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=512)
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def gpu_used_mib():
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return (total_bytes - free_bytes) / 2**20


def run_benchmark(infer_once, cpu_inputs, warmup, iters, baseline_gpu_mib):
    torch.cuda.synchronize()
    started = time.perf_counter()
    mask = infer_once(cpu_inputs[0])
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - started) * 1000.0
    checksum = int(mask.sum(dtype=torch.int64))

    for index in range(warmup):
        infer_once(cpu_inputs[index % len(cpu_inputs)])
    torch.cuda.synchronize()

    latencies = []
    for index in range(iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        mask = infer_once(cpu_inputs[index % len(cpu_inputs)])
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        checksum += int(mask.sum(dtype=torch.int64))

    steady_gpu_mib = max(0.0, gpu_used_mib() - baseline_gpu_mib)
    mean_ms = statistics.fmean(latencies)
    return {
        "first_ms": first_ms,
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(latencies),
        "p90_ms": percentile(latencies, 0.90),
        "images_s": 1000.0 / mean_ms,
        "gpu_delta_mib": steady_gpu_mib,
        "checksum": checksum,
    }


def benchmark_onnxruntime(onnx_path, cpu_inputs, args, ort):
    torch.cuda.empty_cache()
    baseline_gpu_mib = gpu_used_mib()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider_options = {
        "device_id": 0,
        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
    }

    started = time.perf_counter()
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=[("CUDAExecutionProvider", provider_options)],
    )
    startup_ms = (time.perf_counter() - started) * 1000.0
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "ONNX Runtime CUDA provider failed to initialize: "
            + ", ".join(session.get_providers())
        )
    session.disable_fallback()

    ort_output = session.get_outputs()[0]
    output_types = {
        "tensor(float16)": (np.float16, torch.float16),
        "tensor(float)": (np.float32, torch.float32),
    }
    if ort_output.type not in output_types:
        raise TypeError(f"Unsupported ONNX output type: {ort_output.type}")
    numpy_dtype, torch_dtype = output_types[ort_output.type]
    output_shape = tuple(int(dimension) for dimension in ort_output.shape)
    expected_shape = (1, 4, args.input_size, args.input_size)
    if output_shape != expected_shape:
        raise RuntimeError(
            f"Expected ONNX output {expected_shape}, found {output_shape}"
        )

    rgb_cuda = torch.empty_like(cpu_inputs[0], device="cuda")
    logits_cuda = torch.empty(output_shape, dtype=torch_dtype, device="cuda")
    binding = session.io_binding()
    binding.bind_input(
        "rgb_images",
        "cuda",
        0,
        np.float32,
        tuple(rgb_cuda.shape),
        rgb_cuda.data_ptr(),
    )
    binding.bind_output(
        ort_output.name,
        "cuda",
        0,
        numpy_dtype,
        output_shape,
        logits_cuda.data_ptr(),
    )

    def infer_once(cpu_input):
        rgb_cuda.copy_(cpu_input)
        session.run_with_iobinding(binding)
        return logits_cuda.argmax(dim=1).to(torch.uint8).cpu()

    result = run_benchmark(
        infer_once, cpu_inputs, args.warmup, args.iters, baseline_gpu_mib
    )
    result.update(
        backend="ONNXRuntime-CUDA",
        startup_ms=startup_ms,
        artifact_mib=onnx_path.stat().st_size / 2**20,
    )

    del binding, logits_cuda, rgb_cuda, session
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def benchmark_tensorrt(engine_path, cpu_inputs, args, trt):
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("mode") != "fp32":
        raise RuntimeError(
            "Phase 18 only accepts the validated strict FP32 TensorRT engine"
        )

    torch.cuda.empty_cache()
    baseline_gpu_mib = gpu_used_mib()
    started = time.perf_counter()
    runner = TensorRTRunner(engine_path, trt)
    startup_ms = (time.perf_counter() - started) * 1000.0

    rgb_cuda = torch.empty_like(cpu_inputs[0], device="cuda")
    logits_cuda = runner.make_output(rgb_cuda)

    def infer_once(cpu_input):
        rgb_cuda.copy_(cpu_input)
        runner.infer(rgb_cuda, logits_cuda)
        return logits_cuda.argmax(dim=1).to(torch.uint8).cpu()

    result = run_benchmark(
        infer_once, cpu_inputs, args.warmup, args.iters, baseline_gpu_mib
    )
    result.update(
        backend="TensorRT-FP32",
        startup_ms=startup_ms,
        artifact_mib=engine_path.stat().st_size / 2**20,
    )

    del logits_cuda, rgb_cuda, runner
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def print_results(rows, args):
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | Batch: 1 | "
        f"Input: {args.input_size}x{args.input_size} | "
        f"Warmup: {args.warmup} | Iterations: {args.iters}"
    )
    print(
        "Scope: preprocessed CPU float32 RGB tensor -> CPU uint8 segmentation mask "
        "(H2D + inference + GPU argmax + D2H); image decode/resize excluded."
    )
    print()
    print(
        "| Backend          | Artifact(MiB) | Startup(ms) | First(ms) | "
        "Mean(ms) | P50(ms) | P90(ms) | Img/s | GPU Delta(MiB) |"
    )
    print(
        "|------------------|--------------:|------------:|----------:|"
        "---------:|--------:|--------:|------:|---------------:|"
    )
    for row in rows:
        print(
            f"| {row['backend']:<16} | {row['artifact_mib']:13.2f} | "
            f"{row['startup_ms']:11.2f} | {row['first_ms']:9.3f} | "
            f"{row['mean_ms']:8.3f} | {row['p50_ms']:7.3f} | "
            f"{row['p90_ms']:7.3f} | {row['images_s']:5.2f} | "
            f"{row['gpu_delta_mib']:14.1f} |"
        )
    print()
    print(
        "Checksums (anti-lazy-execution only): "
        + ", ".join(f"{row['backend']}={row['checksum']}" for row in rows)
    )
    ort_row, trt_row = rows
    speedup = ort_row["mean_ms"] / trt_row["mean_ms"]
    latency_reduction = 100.0 * (
        ort_row["mean_ms"] - trt_row["mean_ms"]
    ) / ort_row["mean_ms"]
    print(
        f"TensorRT vs ORT end-to-end: {speedup:.3f}x speedup, "
        f"{latency_reduction:+.2f}% latency reduction"
    )


def main():
    args = parse_args()
    if (
        args.samples <= 0
        or args.warmup < 0
        or args.iters <= 0
        or args.input_size <= 0
    ):
        raise ValueError(
            "samples/iters/input-size must be positive and warmup non-negative"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    try:
        import onnxruntime as ort
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "ONNX Runtime GPU and TensorRT 10.x are required"
        ) from error
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
    if int(trt.__version__.split(".", maxsplit=1)[0]) != 10:
        raise RuntimeError(f"TensorRT 10.x is required, found {trt.__version__}")

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    for artifact in (onnx_path, engine_path):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)

    image_paths = list_images(Path(args.image_dir), args.samples)
    cpu_inputs = [
        torch.from_numpy(load_rgb_image(path, args.input_size))
        for path in image_paths
    ]
    torch.backends.cudnn.benchmark = True
    torch.cuda.init()

    rows = [
        benchmark_onnxruntime(onnx_path, cpu_inputs, args, ort),
        benchmark_tensorrt(engine_path, cpu_inputs, args, trt),
    ]
    print_results(rows, args)


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
