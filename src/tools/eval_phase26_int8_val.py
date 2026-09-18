import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_phase17_full_deployment import (  # noqa: E402
    CLASS_NAMES,
    calculate_metrics,
    load_annotation,
    update_confusion,
)
from validate_phase12_onnxruntime import (  # noqa: E402
    CORE_COMPUTE_OPS,
    list_images,
    load_rgb_image,
    provider_breakdown,
    summarize_latency,
)
from validate_phase12_tensorrt import (  # noqa: E402
    TensorRTRunner,
)


BACKENDS = ("ONNX-FP32", "ONNX-QDQ-INT8", "TensorRT-FP16", "TensorRT-INT8")
PHASE26_CORE_COMPUTE_OPS = CORE_COMPUTE_OPS | {
    "ConvInteger",
    "MatMulInteger",
    "QLinearConv",
    "QLinearMatMul",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-validation-set Phase 26 Q/DQ and TensorRT INT8 gate."
    )
    parser.add_argument("--reference-onnx", required=True)
    parser.add_argument("--qdq-onnx", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--expected-samples", type=int, default=535)
    parser.add_argument("--expected-reference-miou", type=float, required=True)
    parser.add_argument("--max-reference-miou-delta", type=float, default=0.05)
    parser.add_argument("--max-quantized-miou-drop", type=float, default=0.50)
    parser.add_argument("--min-quantized-agreement", type=float, default=98.50)
    parser.add_argument("--min-int8-speedup", type=float, default=1.15)
    parser.add_argument("--benchmark-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def make_session(path, ort, profile_prefix=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDAExecutionProvider failed for {path}")
    session.disable_fallback()
    return session


def json_metrics(metrics):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in metrics.items()
    }


def benchmark_tensorrt_rotating(runner, inputs, warmup, iters):
    outputs = [runner.make_output(value) for value in inputs]
    for index in range(warmup):
        slot = index % len(inputs)
        runner.infer(inputs[slot], outputs[slot])
    torch.cuda.synchronize()
    latencies = []
    for index in range(iters):
        slot = index % len(inputs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        runner.infer(inputs[slot], outputs[slot])
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def main():
    args = parse_args()
    if args.input_size <= 0 or args.expected_samples <= 0:
        raise ValueError("input-size and expected-samples must be positive")
    if args.benchmark_samples <= 0 or args.warmup < 0 or args.iters <= 0:
        raise ValueError("invalid benchmark-samples/warmup/iters")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 26")
    try:
        import onnxruntime as ort
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "Phase 26 requires onnxruntime-gpu==1.18.0 and tensorrt-cu12==10.0.1"
        ) from error

    paths = {
        "reference": Path(args.reference_onnx),
        "qdq": Path(args.qdq_onnx),
        "fp16": Path(args.fp16_engine),
        "int8": Path(args.int8_engine),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))
    for key, expected_mode in (("fp16", "mixed-fp16"), ("int8", "int8-qdq")):
        metadata_path = paths[key].with_suffix(paths[key].suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("mode") != expected_mode:
            raise RuntimeError(
                f"Expected {expected_mode} metadata for {paths[key]}, got {metadata.get('mode')}"
            )

    image_dir = Path(args.image_dir)
    ann_dir = Path(args.ann_dir)
    image_paths = list_images(image_dir, 2**31 - 1)
    if len(image_paths) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} validation images, found {len(image_paths)}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_session = make_session(paths["reference"], ort)
    qdq_session = make_session(paths["qdq"], ort)
    fp16_runner = TensorRTRunner(paths["fp16"], trt)
    int8_runner = TensorRTRunner(paths["int8"], trt)

    confusions = {
        name: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        for name in BACKENDS
    }
    nonfinite = {name: 0 for name in BACKENDS}
    agreements = {name: 0 for name in BACKENDS[1:]}
    compared_pixels = 0
    benchmark_inputs = []
    first_rgb_numpy = None

    for index, image_path in enumerate(image_paths, start=1):
        rgb_numpy = load_rgb_image(image_path, args.input_size)
        if first_rgb_numpy is None:
            first_rgb_numpy = rgb_numpy
        rgb_cuda = torch.from_numpy(rgb_numpy).cuda()
        if len(benchmark_inputs) < args.benchmark_samples:
            benchmark_inputs.append(rgb_cuda)
        fp16_output = fp16_runner.make_output(rgb_cuda)
        int8_output = int8_runner.make_output(rgb_cuda)
        reference_logits = reference_session.run(
            None, {"rgb_images": rgb_numpy}
        )[0].astype(np.float32)
        qdq_logits = qdq_session.run(None, {"rgb_images": rgb_numpy})[0].astype(
            np.float32
        )
        with torch.inference_mode():
            fp16_logits = (
                fp16_runner.infer(rgb_cuda, fp16_output).float().cpu().numpy()
            )
            int8_logits = (
                int8_runner.infer(rgb_cuda, int8_output).float().cpu().numpy()
            )
        logits_by_backend = {
            "ONNX-FP32": reference_logits,
            "ONNX-QDQ-INT8": qdq_logits,
            "TensorRT-FP16": fp16_logits,
            "TensorRT-INT8": int8_logits,
        }
        annotation = load_annotation(ann_dir / image_path.relative_to(image_dir))
        valid = (annotation >= 0) & (annotation < len(CLASS_NAMES))
        reference_prediction = reference_logits.argmax(axis=1)[0]
        compared_pixels += int(valid.sum())
        for name, logits in logits_by_backend.items():
            nonfinite[name] += int((~np.isfinite(logits)).sum())
            prediction = logits.argmax(axis=1)[0].astype(np.int64)
            update_confusion(confusions[name], prediction, annotation)
            if name != "ONNX-FP32":
                agreements[name] += int((prediction[valid] == reference_prediction[valid]).sum())
        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"Evaluated {index}/{len(image_paths)}: {image_path.name}")

    provider_counts = {}
    core_cpu_ops = {}
    with tempfile.TemporaryDirectory(prefix="phase26_ort_profiles_") as temporary:
        profile_root = Path(temporary)
        for name, key in (("ONNX-FP32", "reference"), ("ONNX-QDQ-INT8", "qdq")):
            profile_session = make_session(
                paths[key], ort, profile_root / f"{key}_profile"
            )
            profile_session.run(None, {"rgb_images": first_rgb_numpy})
            profile = Path(profile_session.end_profiling())
            counts, cpu_ops = provider_breakdown(profile)
            provider_counts[name] = counts
            core_cpu_ops[name] = sorted(
                set(cpu_ops) & PHASE26_CORE_COMPUTE_OPS
            )

    latency_rows = {}
    for name, values in (
        (
            "TensorRT-FP16",
            benchmark_tensorrt_rotating(
                fp16_runner,
                benchmark_inputs,
                args.warmup,
                args.iters,
            ),
        ),
        (
            "TensorRT-INT8",
            benchmark_tensorrt_rotating(
                int8_runner,
                benchmark_inputs,
                args.warmup,
                args.iters,
            ),
        ),
    ):
        _, mean_ms, median_ms, p90_ms, throughput = summarize_latency(name, values)
        latency_rows[name] = {
            "mean_ms": mean_ms,
            "p50_ms": median_ms,
            "p90_ms": p90_ms,
            "throughput_images_per_second": throughput,
        }

    metrics = {name: calculate_metrics(confusion) for name, confusion in confusions.items()}
    agreement_percent = {
        name: 100.0 * count / compared_pixels for name, count in agreements.items()
    }
    reference_miou = metrics["ONNX-FP32"]["mIoU"]
    quantized_drops = {
        name: reference_miou - metrics[name]["mIoU"] for name in BACKENDS[1:]
    }
    speedup = (
        latency_rows["TensorRT-FP16"]["mean_ms"]
        / latency_rows["TensorRT-INT8"]["mean_ms"]
    )
    gates = {
        "reference_matches_phase25": abs(
            reference_miou - args.expected_reference_miou
        )
        <= args.max_reference_miou_delta,
        "qdq_miou_drop": quantized_drops["ONNX-QDQ-INT8"]
        <= args.max_quantized_miou_drop,
        "fp16_engine_miou_delta": abs(quantized_drops["TensorRT-FP16"])
        <= args.max_reference_miou_delta,
        "int8_engine_miou_drop": quantized_drops["TensorRT-INT8"]
        <= args.max_quantized_miou_drop,
        "quantized_agreement": all(
            agreement_percent[name] >= args.min_quantized_agreement
            for name in ("ONNX-QDQ-INT8", "TensorRT-INT8")
        ),
        "no_nonfinite": not any(nonfinite.values()),
        "no_core_cpu_fallback": not any(core_cpu_ops.values()),
        "int8_speedup": speedup >= args.min_int8_speedup,
    }
    result = {
        "phase": 26,
        "split": "val",
        "test_evaluated": False,
        "samples": len(image_paths),
        "thresholds": {
            "expected_reference_mIoU": args.expected_reference_miou,
            "max_reference_mIoU_delta": args.max_reference_miou_delta,
            "max_quantized_mIoU_drop": args.max_quantized_miou_drop,
            "min_quantized_agreement": args.min_quantized_agreement,
            "min_int8_speedup": args.min_int8_speedup,
        },
        "metrics": {name: json_metrics(value) for name, value in metrics.items()},
        "mIoU_drop_from_reference": quantized_drops,
        "pixel_agreement_with_reference": agreement_percent,
        "nonfinite_logits": nonfinite,
        "provider_counts": provider_counts,
        "core_cpu_fallback_ops": core_cpu_ops,
        "latency": latency_rows,
        "int8_speedup_over_fp16": speedup,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    print("Phase 26 real INT8 validation gate: PASSED")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
