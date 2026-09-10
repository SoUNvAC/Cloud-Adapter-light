import argparse
import gc
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validate_phase12_onnxruntime import (  # noqa: E402
    list_images,
    load_rgb_image,
)


CLASS_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 19 full-set input-resolution Pareto screen"
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="SIZE=ONNX_PATH",
        help="Repeat for each static ONNX resolution, for example 512=model.onnx",
    )
    parser.add_argument(
        "--image-dir", default="data/cloudsen12_high_l1c/img_dir/test"
    )
    parser.add_argument(
        "--ann-dir", default="data/cloudsen12_high_l1c/ann_dir/test"
    )
    parser.add_argument("--expected-samples", type=int, default=975)
    parser.add_argument("--benchmark-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--baseline-size", type=int, default=512)
    parser.add_argument("--expected-baseline-miou", type=float, default=68.728)
    parser.add_argument("--max-baseline-miou-delta", type=float, default=0.05)
    parser.add_argument("--max-miou-drop", type=float, default=0.50)
    parser.add_argument("--min-speedup", type=float, default=1.15)
    return parser.parse_args()


def parse_model_specs(specs):
    models = {}
    for spec in specs:
        try:
            size_text, path_text = spec.split("=", maxsplit=1)
            size = int(size_text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid --model value {spec!r}; use SIZE=PATH") from error
        if size <= 0:
            raise ValueError(f"Model input size must be positive, found {size}")
        if size in models:
            raise ValueError(f"Duplicate model input size: {size}")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        models[size] = path
    return dict(sorted(models.items(), reverse=True))


def load_annotation(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        annotation = np.array(image)
    if annotation.ndim != 2:
        raise ValueError(f"Expected a single-channel annotation at {path}")
    return annotation.astype(np.int64, copy=False)


def update_confusion(confusion, prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/annotation shape mismatch: {prediction.shape} vs {target.shape}"
        )
    valid = (target >= 0) & (target < len(CLASS_NAMES))
    encoded = len(CLASS_NAMES) * target[valid] + prediction[valid]
    confusion += np.bincount(
        encoded, minlength=len(CLASS_NAMES) ** 2
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))


def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def calculate_metrics(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - true_positive
    iou = safe_divide(true_positive, union)
    accuracy = safe_divide(true_positive, ground_truth)
    precision = safe_divide(true_positive, predicted)
    dice = safe_divide(2.0 * true_positive, ground_truth + predicted)
    return {
        "aAcc": 100.0 * true_positive.sum() / ground_truth.sum(),
        "mIoU": 100.0 * np.nanmean(iou),
        "mAcc": 100.0 * np.nanmean(accuracy),
        "mDice": 100.0 * np.nanmean(dice),
        "mPrecision": 100.0 * np.nanmean(precision),
        "mRecall": 100.0 * np.nanmean(accuracy),
        "class_iou": 100.0 * iou,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def create_session(onnx_path, ort):
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
    return session, startup_ms


def evaluate_full_set(session, image_paths, image_dir, ann_dir, input_size):
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    nonfinite = 0
    with torch.inference_mode():
        for image_path in image_paths:
            relative_path = image_path.relative_to(image_dir)
            annotation = load_annotation(ann_dir / relative_path)
            rgb_numpy = load_rgb_image(image_path, input_size)
            logits = session.run(None, {"rgb_images": rgb_numpy})[0]
            nonfinite += int((~np.isfinite(logits)).sum())
            logits_tensor = torch.from_numpy(logits).float()
            if tuple(logits_tensor.shape[-2:]) != tuple(annotation.shape):
                logits_tensor = F.interpolate(
                    logits_tensor,
                    size=annotation.shape,
                    mode="bilinear",
                    align_corners=False,
                )
            prediction = logits_tensor.argmax(dim=1)[0].numpy().astype(np.int64)
            update_confusion(confusion, prediction, annotation)
    if nonfinite:
        raise RuntimeError(
            f"Input {input_size}: ONNX Runtime produced {nonfinite} non-finite logits"
        )
    return calculate_metrics(confusion)


def benchmark_e2e(
    session, image_paths, input_size, output_size, warmup, iters
):
    ort_output = session.get_outputs()[0]
    output_types = {
        "tensor(float16)": (np.float16, torch.float16),
        "tensor(float)": (np.float32, torch.float32),
    }
    if ort_output.type not in output_types:
        raise TypeError(f"Unsupported ONNX output type: {ort_output.type}")
    numpy_dtype, torch_dtype = output_types[ort_output.type]
    output_shape = (1, len(CLASS_NAMES), input_size, input_size)
    reported_shape = tuple(ort_output.shape)
    if len(reported_shape) != len(output_shape):
        raise RuntimeError(f"Unexpected ONNX output shape: {reported_shape}")
    for index, (reported, expected) in enumerate(zip(reported_shape, output_shape)):
        if isinstance(reported, int) and reported != expected:
            raise RuntimeError(
                f"ONNX output dimension {index} is {reported}, expected {expected}"
            )

    cpu_inputs = [
        torch.from_numpy(load_rgb_image(path, input_size)) for path in image_paths
    ]
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
        output = logits_cuda
        if input_size != output_size:
            output = F.interpolate(
                output.float(),
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )
        return output.argmax(dim=1).to(torch.uint8).cpu()

    for index in range(warmup):
        infer_once(cpu_inputs[index % len(cpu_inputs)])
    torch.cuda.synchronize()
    latencies = []
    checksum = 0
    for index in range(iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        mask = infer_once(cpu_inputs[index % len(cpu_inputs)])
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        checksum += int(mask.sum(dtype=torch.int64))
    mean_ms = statistics.fmean(latencies)
    return {
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(latencies),
        "p90_ms": percentile(latencies, 0.90),
        "images_s": 1000.0 / mean_ms,
        "checksum": checksum,
    }


def print_results(results, baseline_size, max_miou_drop, min_speedup):
    baseline = results[baseline_size]
    baseline_miou = baseline["metrics"]["mIoU"]
    baseline_latency = baseline["benchmark"]["mean_ms"]

    print()
    print("Full-set accuracy (975 images):")
    print()
    print("| Input |   aAcc |   mIoU |  Delta |    mAcc |  mDice | mPrecision | mRecall |")
    print("|------:|-------:|-------:|-------:|--------:|-------:|-----------:|--------:|")
    for size, result in results.items():
        metrics = result["metrics"]
        delta = metrics["mIoU"] - baseline_miou
        print(
            f"| {size:>4} | {metrics['aAcc']:6.3f} | {metrics['mIoU']:6.3f} | "
            f"{delta:+6.3f} | {metrics['mAcc']:7.3f} | {metrics['mDice']:6.3f} | "
            f"{metrics['mPrecision']:10.3f} | {metrics['mRecall']:7.3f} |"
        )

    print()
    sizes = list(results)
    print("| Class        | " + " | ".join(f"IoU@{size}" for size in sizes) + " |")
    print("|--------------|" + "|".join("--------:" for _ in sizes) + "|")
    for class_index, class_name in enumerate(CLASS_NAMES):
        values = " | ".join(
            f"{results[size]['metrics']['class_iou'][class_index]:7.3f}"
            for size in sizes
        )
        print(f"| {class_name:<12} | {values} |")

    print()
    print("End-to-end ORT CUDA (CPU tensor -> 512x512 CPU mask):")
    print()
    print(
        "| Input | Tokens | Artifact(MiB) | Startup(ms) | Mean(ms) | "
        "P50(ms) | P90(ms) | Img/s | Speedup | Status |"
    )
    print(
        "|------:|-------:|--------------:|------------:|---------:|"
        "--------:|--------:|------:|--------:|:-------|"
    )
    passed_sizes = []
    for size, result in results.items():
        metrics = result["metrics"]
        benchmark = result["benchmark"]
        miou_drop = baseline_miou - metrics["mIoU"]
        speedup = baseline_latency / benchmark["mean_ms"]
        if size == baseline_size:
            status = "BASE"
        elif miou_drop <= max_miou_drop and speedup >= min_speedup:
            status = "PASS"
            passed_sizes.append(size)
        else:
            status = "FAIL"
        token_side = size // 16
        print(
            f"| {size:>4} | {token_side**2:>6} | {result['artifact_mib']:13.2f} | "
            f"{result['startup_ms']:11.2f} | {benchmark['mean_ms']:8.3f} | "
            f"{benchmark['p50_ms']:7.3f} | {benchmark['p90_ms']:7.3f} | "
            f"{benchmark['images_s']:5.2f} | {speedup:7.3f}x | {status:<6} |"
        )

    if passed_sizes:
        selected = min(passed_sizes, key=lambda size: results[size]["benchmark"]["mean_ms"])
        print(f"Resolution gate: PASSED; fastest eligible candidate is {selected}x{selected}")
    else:
        print("Resolution gate: no lower-resolution candidate passed")
    print(
        "Checksums (anti-lazy only): "
        + ", ".join(
            f"{size}={result['benchmark']['checksum']}"
            for size, result in results.items()
        )
    )


def main():
    args = parse_args()
    if (
        args.expected_samples <= 0
        or args.benchmark_samples <= 0
        or args.warmup < 0
        or args.iters <= 0
        or args.output_size <= 0
        or args.max_baseline_miou_delta < 0
        or args.max_miou_drop < 0
        or args.min_speedup <= 0
    ):
        raise ValueError("Invalid evaluation or benchmark arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX Runtime GPU is required") from error
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")

    models = parse_model_specs(args.model)
    if args.baseline_size not in models:
        raise ValueError(f"Missing baseline model at input size {args.baseline_size}")
    image_dir = Path(args.image_dir)
    ann_dir = Path(args.ann_dir)
    image_paths = list_images(image_dir, 2**31 - 1)
    if len(image_paths) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} test images, found {len(image_paths)}"
        )
    benchmark_paths = image_paths[: args.benchmark_samples]
    torch.cuda.init()

    results = {}
    for size, onnx_path in models.items():
        session, startup_ms = create_session(onnx_path, ort)
        metrics = evaluate_full_set(
            session, image_paths, image_dir, ann_dir, size
        )
        benchmark = benchmark_e2e(
            session,
            benchmark_paths,
            size,
            args.output_size,
            args.warmup,
            args.iters,
        )
        results[size] = {
            "metrics": metrics,
            "benchmark": benchmark,
            "startup_ms": startup_ms,
            "artifact_mib": onnx_path.stat().st_size / 2**20,
        }
        print(f"Evaluated {size}x{size}: {len(image_paths)}/{len(image_paths)}")
        del session
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    measured_baseline_miou = results[args.baseline_size]["metrics"]["mIoU"]
    baseline_delta = abs(measured_baseline_miou - args.expected_baseline_miou)
    if baseline_delta > args.max_baseline_miou_delta:
        raise RuntimeError(
            f"Baseline mIoU {measured_baseline_miou:.5f} differs from expected "
            f"{args.expected_baseline_miou:.5f} by {baseline_delta:.5f}, exceeding "
            f"{args.max_baseline_miou_delta:.5f}"
        )
    print_results(
        results, args.baseline_size, args.max_miou_drop, args.min_speedup
    )


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
