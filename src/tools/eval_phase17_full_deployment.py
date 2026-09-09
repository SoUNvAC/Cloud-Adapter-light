import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from validate_phase12_onnxruntime import list_images, load_rgb_image  # noqa: E402
from validate_phase12_tensorrt import TensorRTRunner  # noqa: E402


CLASS_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-test-set deployment accuracy evaluation for Phase 12"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-samples", type=int, default=975)
    parser.add_argument("--expected-pytorch-miou", type=float, default=68.73)
    parser.add_argument("--max-reference-miou-delta", type=float, default=0.05)
    parser.add_argument("--min-agreement", type=float, default=99.9)
    parser.add_argument("--max-miou-delta", type=float, default=0.05)
    return parser.parse_args()


def load_annotation(path: Path):
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
    return int(valid.sum())


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


def print_results(metrics_by_backend):
    print()
    print("| Backend          |   aAcc |   mIoU |    mAcc |  mDice | mPrecision | mRecall |")
    print("|------------------|-------:|-------:|--------:|-------:|-----------:|--------:|")
    for name, metrics in metrics_by_backend.items():
        print(
            f"| {name:<16} | {metrics['aAcc']:6.3f} | "
            f"{metrics['mIoU']:6.3f} | {metrics['mAcc']:7.3f} | "
            f"{metrics['mDice']:6.3f} | {metrics['mPrecision']:10.3f} | "
            f"{metrics['mRecall']:7.3f} |"
        )

    print()
    print("| Class        | PyTorch IoU | ORT IoU | TensorRT IoU |")
    print("|--------------|------------:|--------:|-------------:|")
    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"| {class_name:<12} | "
            f"{metrics_by_backend['PyTorch-FP16']['class_iou'][index]:11.3f} | "
            f"{metrics_by_backend['ONNXRuntime-CUDA']['class_iou'][index]:7.3f} | "
            f"{metrics_by_backend['TensorRT-FP32']['class_iou'][index]:12.3f} |"
        )


def main():
    args = parse_args()
    if args.input_size <= 0 or args.limit < 0 or args.expected_samples <= 0:
        raise ValueError(
            "input-size and expected-samples must be positive; limit must be non-negative"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    try:
        import onnxruntime as ort
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("ONNX Runtime GPU and TensorRT are required") from error

    image_dir = Path(args.image_dir)
    ann_dir = Path(args.ann_dir)
    limit = args.limit if args.limit else 2**31 - 1
    image_paths = list_images(image_dir, limit)
    if not image_paths:
        raise FileNotFoundError(image_dir)
    if args.limit == 0 and len(image_paths) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} test images, found {len(image_paths)}"
        )

    engine_path = Path(args.engine)
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"TensorRT engine metadata is required to verify FP32 mode: {metadata_path}"
        )
    engine_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    engine_mode = engine_metadata.get("mode")
    if engine_mode != "fp32":
        raise RuntimeError(
            f"Phase 17 requires the strict FP32 engine, found mode={engine_mode!r}"
        )

    wrapper_args = argparse.Namespace(
        config=args.config,
        checkpoint=args.checkpoint,
        precision="fp16",
    )
    wrapper, _ = build_wrapper(wrapper_args)
    torch.backends.cudnn.benchmark = True

    ort_options = ort.SessionOptions()
    ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        args.onnx,
        sess_options=ort_options,
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )
    if "CUDAExecutionProvider" not in ort_session.get_providers():
        raise RuntimeError("ONNX Runtime CUDA provider failed to initialize")
    ort_session.disable_fallback()

    trt_runner = TensorRTRunner(engine_path, trt)
    confusions = {
        name: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        for name in ("PyTorch-FP16", "ONNXRuntime-CUDA", "TensorRT-FP32")
    }
    agreement = {"PyTorch/ORT": 0, "PyTorch/TensorRT": 0, "ORT/TensorRT": 0}
    compared_pixels = 0
    nonfinite = {"PyTorch-FP16": 0, "ONNXRuntime-CUDA": 0, "TensorRT-FP32": 0}

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(image_dir)
        annotation = load_annotation(ann_dir / relative_path)
        rgb_numpy = load_rgb_image(image_path, args.input_size)
        rgb_cuda = torch.from_numpy(rgb_numpy).cuda()
        trt_output = trt_runner.make_output(rgb_cuda)

        with torch.inference_mode():
            pytorch_logits = wrapper(rgb_cuda).float().cpu().numpy()
            trt_logits = (
                trt_runner.infer(rgb_cuda, trt_output).float().cpu().numpy()
            )
        ort_logits = ort_session.run(None, {"rgb_images": rgb_numpy})[0].astype(
            np.float32
        )

        logits_by_backend = {
            "PyTorch-FP16": pytorch_logits,
            "ONNXRuntime-CUDA": ort_logits,
            "TensorRT-FP32": trt_logits,
        }
        predictions = {}
        for name, logits in logits_by_backend.items():
            nonfinite[name] += int((~np.isfinite(logits)).sum())
            predictions[name] = logits.argmax(axis=1)[0].astype(np.int64)
            update_confusion(confusions[name], predictions[name], annotation)

        valid = (annotation >= 0) & (annotation < len(CLASS_NAMES))
        compared_pixels += int(valid.sum())
        agreement["PyTorch/ORT"] += int(
            (
                predictions["PyTorch-FP16"][valid]
                == predictions["ONNXRuntime-CUDA"][valid]
            ).sum()
        )
        agreement["PyTorch/TensorRT"] += int(
            (
                predictions["PyTorch-FP16"][valid]
                == predictions["TensorRT-FP32"][valid]
            ).sum()
        )
        agreement["ORT/TensorRT"] += int(
            (
                predictions["ONNXRuntime-CUDA"][valid]
                == predictions["TensorRT-FP32"][valid]
            ).sum()
        )

        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"Evaluated {index}/{len(image_paths)}: {image_path.name}")

    metrics_by_backend = {
        name: calculate_metrics(confusion) for name, confusion in confusions.items()
    }
    print_results(metrics_by_backend)

    print("\nFull-set pixel agreement:")
    agreement_percent = {}
    for name, count in agreement.items():
        agreement_percent[name] = 100.0 * count / compared_pixels
        print(f"- {name}: {agreement_percent[name]:.5f}%")
    print(
        "Non-finite logits: "
        + ", ".join(f"{name}={count}" for name, count in nonfinite.items())
    )

    if any(nonfinite.values()):
        raise RuntimeError(f"Non-finite logits detected: {nonfinite}")
    for name in ("PyTorch/ORT", "PyTorch/TensorRT"):
        if agreement_percent[name] < args.min_agreement:
            raise RuntimeError(
                f"{name} agreement {agreement_percent[name]:.5f}% is below "
                f"{args.min_agreement:.5f}%"
            )
    reference_miou = metrics_by_backend["PyTorch-FP16"]["mIoU"]
    reference_delta = abs(reference_miou - args.expected_pytorch_miou)
    if reference_delta > args.max_reference_miou_delta:
        raise RuntimeError(
            f"PyTorch mIoU {reference_miou:.5f} differs from expected "
            f"{args.expected_pytorch_miou:.5f} by {reference_delta:.5f}, exceeding "
            f"{args.max_reference_miou_delta:.5f}"
        )
    for name in ("ONNXRuntime-CUDA", "TensorRT-FP32"):
        delta = abs(metrics_by_backend[name]["mIoU"] - reference_miou)
        if delta > args.max_miou_delta:
            raise RuntimeError(
                f"{name} mIoU delta {delta:.5f} exceeds {args.max_miou_delta:.5f}"
            )
    print("Full deployment accuracy gate: PASSED")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
