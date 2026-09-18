import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image

from phase30_jetson_runtime import (
    StandaloneTensorRTRunner,
    TegrastatsMonitor,
    percentile,
    summarize_power,
)


CLASS_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-val accuracy, latency, and energy gate on Jetson Orin Nano."
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=535)
    parser.add_argument("--benchmark-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--tegrastats-interval-ms", type=int, default=100)
    parser.add_argument("--max-miou-drop", type=float, default=0.50)
    parser.add_argument("--max-mean-latency-ms", type=float, default=80.0)
    parser.add_argument("--max-p90-latency-ms", type=float, default=100.0)
    parser.add_argument("--max-repeat-cv-percent", type=float, default=5.0)
    parser.add_argument("--max-energy-j-per-image", type=float, default=2.0)
    parser.add_argument("--max-peak-power-w", type=float, default=25.0)
    parser.add_argument("--max-temperature-c", type=float, default=80.0)
    parser.add_argument("--max-ram-mb", type=float, default=7500.0)
    return parser.parse_args()


def list_images(root):
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images in {root}")
    return paths


def load_rgb(path, size=512):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def load_annotation(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        target = np.asarray(image)
    if target.ndim != 2:
        raise ValueError(f"Expected one-channel annotation: {path}")
    return target.astype(np.int64, copy=False)


def update_confusion(confusion, prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
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
    target = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = target + predicted - true_positive
    iou = safe_divide(true_positive, union)
    recall = safe_divide(true_positive, target)
    precision = safe_divide(true_positive, predicted)
    dice = safe_divide(2.0 * true_positive, target + predicted)
    return {
        "aAcc": 100.0 * true_positive.sum() / target.sum(),
        "mIoU": 100.0 * float(np.nanmean(iou)),
        "mAcc": 100.0 * float(np.nanmean(recall)),
        "mDice": 100.0 * float(np.nanmean(dice)),
        "mPrecision": 100.0 * float(np.nanmean(precision)),
        "class_iou": {
            name: 100.0 * float(iou[index]) for index, name in enumerate(CLASS_NAMES)
        },
    }


def main():
    args = parse_args()
    positive = (
        args.expected_samples,
        args.benchmark_samples,
        args.iters,
        args.benchmark_repeats,
        args.idle_seconds,
        args.tegrastats_interval_ms,
    )
    if any(value <= 0 for value in positive) or args.warmup < 0:
        raise ValueError("sample/iteration/idle arguments must be positive")
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("Phase 30 Jetson environment audit did not pass")
    if selection.get("test_evaluated") is not False:
        raise RuntimeError("Selected Phase 25 checkpoint must keep test sealed")
    expected_miou = float(selection["reference_validation_mIoU"])
    engine_path = Path(args.engine)
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("device_model") != audit.get("device_model"):
        raise RuntimeError("Engine was not built on the audited target device")
    if metadata.get("engine_sha256") != sha256_file(engine_path):
        raise RuntimeError("Engine hash does not match target-local build metadata")

    image_dir = Path(args.image_dir)
    ann_dir = Path(args.ann_dir)
    image_paths = list_images(image_dir)
    if len(image_paths) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} validation images, found {len(image_paths)}"
        )
    runner = StandaloneTensorRTRunner(engine_path)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    invalid_predictions = 0
    benchmark_inputs = []
    try:
        for index, image_path in enumerate(image_paths, start=1):
            rgb = load_rgb(image_path)
            if len(benchmark_inputs) < args.benchmark_samples:
                benchmark_inputs.append(rgb)
            prediction = runner.infer(rgb)[0].astype(np.int64, copy=False)
            invalid_predictions += int(((prediction < 0) | (prediction > 3)).sum())
            target = load_annotation(ann_dir / image_path.relative_to(image_dir))
            update_confusion(confusion, prediction, target)
            if index == 1 or index % 50 == 0 or index == len(image_paths):
                print(f"Evaluated {index}/{len(image_paths)}: {image_path.name}")

        started = time.perf_counter()
        first_mask = runner.infer(benchmark_inputs[0])
        first_ms = (time.perf_counter() - started) * 1000.0
        checksum = int(first_mask.sum(dtype=np.int64))

        monitor = TegrastatsMonitor(args.tegrastats_interval_ms)
        monitor.start()
        try:
            time.sleep(args.idle_seconds)
            idle_count = len(monitor.samples)
            for index in range(args.warmup):
                runner.infer(benchmark_inputs[index % len(benchmark_inputs)])
            repeat_means = []
            all_latencies = []
            for repeat in range(args.benchmark_repeats):
                current = []
                for index in range(args.iters):
                    input_array = benchmark_inputs[index % len(benchmark_inputs)]
                    started = time.perf_counter()
                    mask = runner.infer(input_array)
                    elapsed = (time.perf_counter() - started) * 1000.0
                    current.append(elapsed)
                    checksum += int(mask.sum(dtype=np.int64))
                repeat_means.append(statistics.fmean(current))
                all_latencies.extend(current)
                print(
                    f"Benchmark repeat {repeat + 1}/{args.benchmark_repeats}: "
                    f"{repeat_means[-1]:.3f} ms"
                )
        finally:
            monitor.stop()
    finally:
        runner.close()

    mean_ms = statistics.fmean(all_latencies)
    repeat_cv = (
        100.0 * statistics.stdev(repeat_means) / statistics.fmean(repeat_means)
        if len(repeat_means) > 1
        else 0.0
    )
    benchmark = {
        "contract": "preloaded CPU float32 RGB -> CPU uint8 mask",
        "batch_size": 1,
        "input_size": 512,
        "benchmark_samples": len(benchmark_inputs),
        "warmup": args.warmup,
        "iters_per_repeat": args.iters,
        "repeats": args.benchmark_repeats,
        "first_ms": first_ms,
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(all_latencies),
        "p90_ms": percentile(all_latencies, 0.90),
        "p99_ms": percentile(all_latencies, 0.99),
        "throughput_images_s": 1000.0 / mean_ms,
        "repeat_means_ms": repeat_means,
        "repeat_cv_percent": repeat_cv,
        "checksum": checksum,
    }
    power = summarize_power(monitor.samples, idle_count, mean_ms)
    metrics = calculate_metrics(confusion)
    miou_drop = expected_miou - metrics["mIoU"]
    gates = {
        "all_validation_images": len(image_paths) == args.expected_samples,
        "valid_uint8_masks": invalid_predictions == 0,
        "accuracy_preserved": miou_drop <= args.max_miou_drop,
        "mean_latency": benchmark["mean_ms"] <= args.max_mean_latency_ms,
        "p90_latency": benchmark["p90_ms"] <= args.max_p90_latency_ms,
        "repeat_stability": repeat_cv <= args.max_repeat_cv_percent,
        "energy_per_image": power["total_energy_j_per_image"]
        <= args.max_energy_j_per_image,
        "peak_power": power["active_vdd_in_peak_w"] <= args.max_peak_power_w,
        "temperature": power["peak_temperature_c"] is not None
        and power["peak_temperature_c"] <= args.max_temperature_c,
        "memory": power["peak_ram_mb"] is not None
        and power["peak_ram_mb"] <= args.max_ram_mb,
        "anti_lazy_checksum": checksum > 0,
        "pytorch_not_imported": "torch" not in __import__("sys").modules,
    }
    result = {
        "phase": 30,
        "split": "val",
        "cloudsen12_test_evaluated": False,
        "device_audit": audit,
        "engine_metadata": metadata,
        "selection_sha256": sha256_file(Path(args.selection)),
        "expected_reference_mIoU": expected_miou,
        "metrics": metrics,
        "mIoU_drop_from_phase25": miou_drop,
        "confusion": confusion.tolist(),
        "invalid_prediction_pixels": invalid_predictions,
        "benchmark": benchmark,
        "power": power,
        "tegrastats_samples": monitor.samples,
        "thresholds": {
            "max_mIoU_drop": args.max_miou_drop,
            "max_mean_latency_ms": args.max_mean_latency_ms,
            "max_p90_latency_ms": args.max_p90_latency_ms,
            "max_repeat_cv_percent": args.max_repeat_cv_percent,
            "max_energy_j_per_image": args.max_energy_j_per_image,
            "max_peak_power_w": args.max_peak_power_w,
            "max_temperature_c": args.max_temperature_c,
            "max_ram_mb": args.max_ram_mb,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "stop_loss": {
            "on_failure": "do not claim target-edge real-time deployment",
            "next_direction": "reduce FP16 whole-network channels/tokens and repeat the unchanged target-local benchmark",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("metrics", "benchmark", "power", "gates", "passed")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
