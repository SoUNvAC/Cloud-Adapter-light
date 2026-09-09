import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")

try:
    import cv2
except ImportError as error:
    raise RuntimeError(
        "OpenCV is missing. Run: bash tools/install_onnxruntime_cu12.sh"
    ) from error
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Phase 12 ONNX CUDA accuracy and latency"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument(
        "--image-dir", default="data/cloudsen12_high_l1c/img_dir/test"
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--precision", choices=("fp16",), default="fp16")
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def load_rgb_image(path: Path, size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    if bgr.shape[:2] != (size, size):
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32)


def list_images(image_dir: Path, limit: int):
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    return paths[:limit]


def provider_breakdown(profile_path: Path):
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    counts = {}
    cpu_ops = set()
    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args", {})
        provider = args.get("provider")
        if not provider:
            continue
        counts[provider] = counts.get(provider, 0) + 1
        if provider == "CPUExecutionProvider":
            cpu_ops.add(args.get("op_name", event.get("name", "unknown")))
    return counts, sorted(cpu_ops)


CORE_COMPUTE_OPS = {
    "Attention",
    "BatchNormalization",
    "Conv",
    "Einsum",
    "Gemm",
    "InstanceNormalization",
    "LayerNormalization",
    "MatMul",
    "MultiHeadAttention",
    "Resize",
    "Softmax",
}


def benchmark_pytorch(wrapper, rgb_cuda, warmup: int, iters: int):
    with torch.inference_mode():
        for _ in range(warmup):
            output = wrapper(rgb_cuda)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(iters):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = wrapper(rgb_cuda)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
    del output
    return latencies


def benchmark_onnxruntime(session, rgb_cuda, warmup: int, iters: int):
    binding = session.io_binding()
    binding.bind_input(
        "rgb_images",
        "cuda",
        0,
        np.float32,
        tuple(rgb_cuda.shape),
        rgb_cuda.data_ptr(),
    )
    binding.bind_output("seg_logits", "cuda", 0)
    for _ in range(warmup):
        session.run_with_iobinding(binding)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        session.run_with_iobinding(binding)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def summarize_latency(name, latencies):
    mean_ms = statistics.fmean(latencies)
    return (
        name,
        mean_ms,
        statistics.median(latencies),
        percentile(latencies, 0.90),
        1000.0 / mean_ms,
    )


def main():
    args = parse_args()
    if args.samples <= 0 or args.iters <= 0 or args.warmup < 0:
        raise ValueError("samples/iters must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "Install the CUDA 12 GPU runtime with: "
            "bash tools/install_onnxruntime_cu12.sh"
        ) from error
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "ONNX Runtime CUDAExecutionProvider is unavailable. Installed providers: "
            + ", ".join(ort.get_available_providers())
        )

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    image_paths = list_images(Path(args.image_dir), args.samples)

    wrapper_args = argparse.Namespace(
        config=args.config,
        checkpoint=args.checkpoint,
        precision=args.precision,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    torch.backends.cudnn.benchmark = True

    with tempfile.TemporaryDirectory(prefix="cloud_adapter_ort_profile_") as tmp:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(tmp) / "profile")
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": 0})],
        )
        active_providers = session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "CUDAExecutionProvider failed to initialize and ONNX Runtime "
                "fell back to CPU. Active providers: "
                + ", ".join(active_providers)
                + ". Install the CUDA 12/cuDNN 8 wheel with: "
                "bash tools/install_onnxruntime_cu12.sh"
            )
        session.disable_fallback()

        max_abs_error = 0.0
        absolute_error_sum = 0.0
        compared_values = 0
        equal_pixels = 0
        compared_pixels = 0
        first_rgb = None
        for index, image_path in enumerate(image_paths):
            rgb_numpy = load_rgb_image(image_path, args.input_size)
            if first_rgb is None:
                first_rgb = rgb_numpy
            rgb_cuda = torch.from_numpy(rgb_numpy).cuda()
            with torch.inference_mode():
                pytorch_output = wrapper(rgb_cuda).float().cpu().numpy()
            ort_output = session.run(None, {"rgb_images": rgb_numpy})[0].astype(
                np.float32
            )
            error = np.abs(pytorch_output - ort_output)
            max_abs_error = max(max_abs_error, float(error.max()))
            absolute_error_sum += float(error.sum(dtype=np.float64))
            compared_values += error.size
            equal_pixels += int(
                (pytorch_output.argmax(axis=1) == ort_output.argmax(axis=1)).sum()
            )
            compared_pixels += pytorch_output.shape[0] * args.input_size**2
            print(
                f"Validated {index + 1:02d}/{len(image_paths):02d}: "
                f"{image_path.name}",
                end="\r",
                flush=True,
            )
        print()

        profile_path = Path(session.end_profiling())
        providers, cpu_ops = provider_breakdown(profile_path)

        first_rgb_cuda = torch.from_numpy(first_rgb).cuda()
        pytorch_latencies = benchmark_pytorch(
            wrapper, first_rgb_cuda, args.warmup, args.iters
        )
        ort_latencies = benchmark_onnxruntime(
            session, first_rgb_cuda, args.warmup, args.iters
        )

    mean_abs_error = absolute_error_sum / compared_values
    pixel_agreement = 100.0 * equal_pixels / compared_pixels
    print(f"ONNX Runtime: {ort.__version__}")
    print("Providers: " + ", ".join(session.get_providers()))
    print(
        f"Numerical parity ({len(image_paths)} images): "
        f"max_abs={max_abs_error:.6g}, mean_abs={mean_abs_error:.6g}, "
        f"argmax_agreement={pixel_agreement:.5f}%"
    )
    print(
        "Profiled node providers: "
        + ", ".join(f"{name}={count}" for name, count in providers.items())
    )
    if cpu_ops:
        print("CPU-assigned op types: " + ", ".join(cpu_ops))

    rows = [
        summarize_latency("PyTorch-FP16", pytorch_latencies),
        summarize_latency("ONNXRuntime-CUDA", ort_latencies),
    ]
    print()
    print("| Backend          | Mean(ms) | P50(ms) | P90(ms) | Img/s |")
    print("|------------------|---------:|--------:|--------:|------:|")
    for name, mean_ms, median_ms, p90_ms, throughput in rows:
        print(
            f"| {name:<16} | {mean_ms:8.3f} | {median_ms:7.3f} | "
            f"{p90_ms:7.3f} | {throughput:5.2f} |"
        )

    pytorch_mean = rows[0][1]
    ort_mean = rows[1][1]
    speedup = pytorch_mean / ort_mean
    latency_reduction = 100.0 * (pytorch_mean - ort_mean) / pytorch_mean
    print(
        f"\nORT vs PyTorch: {speedup:.3f}x throughput-equivalent speedup, "
        f"{latency_reduction:+.2f}% latency reduction"
    )

    if pixel_agreement < 99.9:
        raise RuntimeError(
            f"Argmax agreement {pixel_agreement:.5f}% is below the 99.9% gate"
        )
    core_cpu_ops = sorted(set(cpu_ops) & CORE_COMPUTE_OPS)
    if core_cpu_ops:
        raise RuntimeError(
            "Core compute operators fell back to CPU: " + ", ".join(core_cpu_ops)
        )
    print("Deployment gate: PASSED")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
