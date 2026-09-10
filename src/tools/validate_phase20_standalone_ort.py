import argparse
import gc
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
from PIL import Image


CLASS_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the standalone uint8-mask ONNX without PyTorch"
    )
    parser.add_argument("--mask-onnx", required=True)
    parser.add_argument("--logits-onnx", required=True)
    parser.add_argument(
        "--image-dir", default="data/cloudsen12_high_l1c/img_dir/test"
    )
    parser.add_argument(
        "--ann-dir", default="data/cloudsen12_high_l1c/ann_dir/test"
    )
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--expected-samples", type=int, default=975)
    parser.add_argument("--parity-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--expected-miou", type=float, default=68.728)
    parser.add_argument("--max-miou-delta", type=float, default=0.05)
    parser.add_argument("--min-agreement", type=float, default=99.99)
    return parser.parse_args()


def list_images(image_dir):
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    return paths


def load_rgb_image(path, size):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    if bgr.shape[:2] != (size, size):
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32)


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


def create_session(path, ort):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )
    startup_ms = (time.perf_counter() - started) * 1000.0
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA provider is not primary: " + ", ".join(providers))
    session.disable_fallback()
    return session, startup_ms


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def benchmark_session_run(session, inputs, warmup, iters):
    def infer_once(input_array):
        return session.run(None, {"rgb_images": input_array})[0]

    return benchmark_callable(infer_once, inputs, warmup, iters, "session.run")


def benchmark_ortvalue_iobinding(session, inputs, warmup, iters, ort, size):
    input_value = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, 3, size, size], np.float32, "cuda", 0
    )
    output_value = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, size, size], np.uint8, "cuda", 0
    )
    binding = session.io_binding()
    binding.bind_ortvalue_input("rgb_images", input_value)
    binding.bind_ortvalue_output("seg_mask", output_value)

    def infer_once(input_array):
        input_value.update_inplace(input_array)
        session.run_with_iobinding(binding)
        return output_value.numpy()

    return benchmark_callable(
        infer_once, inputs, warmup, iters, "OrtValue-I/O-binding"
    )


def benchmark_callable(infer_once, inputs, warmup, iters, mode):
    started = time.perf_counter()
    mask = infer_once(inputs[0])
    first_ms = (time.perf_counter() - started) * 1000.0
    checksum = int(mask.sum(dtype=np.int64))
    for index in range(warmup):
        infer_once(inputs[index % len(inputs)])

    latencies = []
    for index in range(iters):
        started = time.perf_counter()
        mask = infer_once(inputs[index % len(inputs)])
        latencies.append((time.perf_counter() - started) * 1000.0)
        checksum += int(mask.sum(dtype=np.int64))
    mean_ms = statistics.fmean(latencies)
    return {
        "first_ms": first_ms,
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(latencies),
        "p90_ms": percentile(latencies, 0.90),
        "images_s": 1000.0 / mean_ms,
        "checksum": checksum,
        "mode": mode,
    }


def print_results(metrics, benchmark_result, startup_ms, artifact_mib, agreement):
    print()
    print("| Backend             |   aAcc |   mIoU |    mAcc |  mDice | mPrecision | mRecall |")
    print("|---------------------|-------:|-------:|--------:|-------:|-----------:|--------:|")
    print(
        f"| ORT-Standalone-Mask | {metrics['aAcc']:6.3f} | {metrics['mIoU']:6.3f} | "
        f"{metrics['mAcc']:7.3f} | {metrics['mDice']:6.3f} | "
        f"{metrics['mPrecision']:10.3f} | {metrics['mRecall']:7.3f} |"
    )
    print()
    print("| Class        |    IoU |")
    print("|--------------|-------:|")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"| {class_name:<12} | {metrics['class_iou'][index]:6.3f} |")
    print()
    print(
        "| I/O mode | Artifact(MiB) | Startup(ms) | First(ms) | Mean(ms) | "
        "P50(ms) | P90(ms) | Img/s |"
    )
    print(
        "|:---------|--------------:|------------:|----------:|---------:|"
        "--------:|--------:|------:|"
    )
    print(
        f"| {benchmark_result['mode']} | {artifact_mib:13.2f} | "
        f"{startup_ms:11.2f} | "
        f"{benchmark_result['first_ms']:9.3f} | {benchmark_result['mean_ms']:8.3f} | "
        f"{benchmark_result['p50_ms']:7.3f} | {benchmark_result['p90_ms']:7.3f} | "
        f"{benchmark_result['images_s']:5.2f} |"
    )
    print()
    print(f"Mask graph vs logits graph agreement: {agreement:.5f}%")
    print(f"Checksum (anti-lazy only): {benchmark_result['checksum']}")
    print("PyTorch imported by standalone validator: no")


def main():
    args = parse_args()
    if (
        args.input_size <= 0
        or args.expected_samples <= 0
        or args.parity_samples <= 0
        or args.warmup < 0
        or args.iters <= 0
        or args.max_miou_delta < 0
    ):
        raise ValueError("Invalid validation or benchmark arguments")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX Runtime GPU is required") from error
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")

    mask_onnx = Path(args.mask_onnx)
    logits_onnx = Path(args.logits_onnx)
    for artifact in (mask_onnx, logits_onnx):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    image_dir = Path(args.image_dir)
    ann_dir = Path(args.ann_dir)
    image_paths = list_images(image_dir)
    if len(image_paths) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} test images, found {len(image_paths)}"
        )

    mask_session, _ = create_session(mask_onnx, ort)
    logits_session, _ = create_session(logits_onnx, ort)
    mask_output = mask_session.get_outputs()[0]
    if mask_output.name != "seg_mask" or mask_output.type != "tensor(uint8)":
        raise RuntimeError(
            f"Unexpected mask output contract: {mask_output.name}, {mask_output.type}"
        )

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    equal_pixels = 0
    parity_pixels = 0
    for index, image_path in enumerate(image_paths):
        rgb = load_rgb_image(image_path, args.input_size)
        mask = mask_session.run(None, {"rgb_images": rgb})[0]
        expected_shape = (1, args.input_size, args.input_size)
        if mask.dtype != np.uint8 or mask.shape != expected_shape:
            raise RuntimeError(
                f"Invalid output for {image_path}: dtype={mask.dtype}, shape={mask.shape}"
            )
        relative_path = image_path.relative_to(image_dir)
        annotation = load_annotation(ann_dir / relative_path)
        update_confusion(confusion, mask[0].astype(np.int64), annotation)
        if index < args.parity_samples:
            logits = logits_session.run(None, {"rgb_images": rgb})[0]
            reference = logits.argmax(axis=1).astype(np.uint8)
            equal_pixels += int((mask == reference).sum())
            parity_pixels += mask.size

    metrics = calculate_metrics(confusion)
    agreement = 100.0 * equal_pixels / parity_pixels
    del mask_session, logits_session
    gc.collect()

    benchmark_session, startup_ms = create_session(mask_onnx, ort)
    benchmark_inputs = [
        load_rgb_image(path, args.input_size)
        for path in image_paths[: args.parity_samples]
    ]
    benchmark_result = benchmark_ortvalue_iobinding(
        benchmark_session,
        benchmark_inputs,
        args.warmup,
        args.iters,
        ort,
        args.input_size,
    )
    torch_modules = [
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    ]
    if torch_modules:
        raise RuntimeError(
            "Standalone validator unexpectedly imported PyTorch modules: "
            + ", ".join(torch_modules[:10])
        )
    print_results(
        metrics,
        benchmark_result,
        startup_ms,
        mask_onnx.stat().st_size / 2**20,
        agreement,
    )

    miou_delta = abs(metrics["mIoU"] - args.expected_miou)
    if miou_delta > args.max_miou_delta:
        raise RuntimeError(
            f"mIoU {metrics['mIoU']:.5f} differs from expected "
            f"{args.expected_miou:.5f} by {miou_delta:.5f}"
        )
    if agreement < args.min_agreement:
        raise RuntimeError(
            f"Mask/logits agreement {agreement:.5f}% is below "
            f"{args.min_agreement:.5f}%"
        )
    print("Standalone mask deployment gate: PASSED")


if __name__ == "__main__":
    main()
