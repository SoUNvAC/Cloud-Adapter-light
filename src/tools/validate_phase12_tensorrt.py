import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from validate_phase12_onnxruntime import (  # noqa: E402
    benchmark_onnxruntime,
    benchmark_pytorch,
    list_images,
    load_rgb_image,
    summarize_latency,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Phase 12 TensorRT accuracy and latency"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument(
        "--image-dir", default="data/cloudsen12_high_l1c/img_dir/test"
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--min-agreement", type=float, default=99.9)
    return parser.parse_args()


def torch_dtype_for_trt(trt_dtype, trt):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "uint8"):
        mapping[trt.uint8] = torch.uint8
    if hasattr(trt, "int64"):
        mapping[trt.int64] = torch.int64
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    if trt_dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT tensor dtype: {trt_dtype}")
    return mapping[trt_dtype]


class TensorRTRunner:
    def __init__(self, engine_path: Path, trt):
        self.trt = trt
        # TensorRT objects retain the logger internally, so keep it alive for
        # the complete runtime/context lifetime.
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create a TensorRT execution context")
        # enqueueV3 on CUDA's default stream inserts extra global
        # synchronization inside TensorRT. A dedicated stream avoids that
        # warning while explicit stream dependencies preserve correctness.
        self.stream = torch.cuda.Stream(device=torch.cuda.current_device())

        inputs = []
        outputs = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if inputs != ["rgb_images"] or outputs != ["seg_logits"]:
            raise RuntimeError(
                f"Unexpected engine I/O tensors: inputs={inputs}, outputs={outputs}"
            )
        self.input_name = inputs[0]
        self.output_name = outputs[0]

    def make_output(self, input_tensor):
        if not self.context.set_input_shape(self.input_name, tuple(input_tensor.shape)):
            raise RuntimeError(f"TensorRT rejected input shape {tuple(input_tensor.shape)}")
        shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"Unresolved TensorRT output shape: {shape}")
        dtype = torch_dtype_for_trt(
            self.engine.get_tensor_dtype(self.output_name), self.trt
        )
        return torch.empty(shape, dtype=dtype, device=input_tensor.device)

    def infer(self, input_tensor, output_tensor):
        expected_dtype = torch_dtype_for_trt(
            self.engine.get_tensor_dtype(self.input_name), self.trt
        )
        if input_tensor.dtype != expected_dtype or not input_tensor.is_contiguous():
            raise TypeError(
                f"Expected contiguous {expected_dtype} input, got "
                f"{input_tensor.dtype}, contiguous={input_tensor.is_contiguous()}"
            )
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())
        caller_stream = torch.cuda.current_stream(input_tensor.device)
        self.stream.wait_stream(caller_stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        caller_stream.wait_stream(self.stream)
        output_tensor.record_stream(self.stream)
        return output_tensor


def benchmark_tensorrt(runner, rgb_cuda, output, warmup, iters):
    for _ in range(warmup):
        runner.infer(rgb_cuda, output)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        runner.infer(rgb_cuda, output)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def update_parity(reference, candidate, totals):
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    totals["reference_nonfinite"] += int((~reference_finite).sum())
    totals["candidate_nonfinite"] += int((~candidate_finite).sum())
    finite = reference_finite & candidate_finite
    if finite.any():
        error = np.abs(reference[finite] - candidate[finite])
        totals["max_abs"] = max(totals["max_abs"], float(error.max()))
        totals["abs_sum"] += float(error.sum(dtype=np.float64))
        totals["values"] += error.size
    totals["equal"] += int(
        (reference.argmax(axis=1) == candidate.argmax(axis=1)).sum()
    )
    totals["pixels"] += int(np.prod(reference.shape[:1] + reference.shape[2:]))


def parity_summary(totals):
    return (
        totals["max_abs"],
        totals["abs_sum"] / totals["values"] if totals["values"] else float("nan"),
        100.0 * totals["equal"] / totals["pixels"],
        totals["reference_nonfinite"],
        totals["candidate_nonfinite"],
    )


def main():
    args = parse_args()
    if args.samples <= 0 or args.iters <= 0 or args.warmup < 0:
        raise ValueError("samples/iters must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    try:
        import onnxruntime as ort
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "Install onnxruntime-gpu==1.18.0 and tensorrt-cu12==10.0.1"
        ) from error
    if int(trt.__version__.split(".", maxsplit=1)[0]) != 10:
        raise RuntimeError(f"TensorRT 10.x is required, found {trt.__version__}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not engine_path.is_file():
        raise FileNotFoundError(engine_path)
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if metadata_path.is_file():
        engine_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        engine_mode = engine_metadata.get("mode", "unknown")
    else:
        engine_mode = "unknown"
    backend_names = {
        "mixed-fp16": "TensorRT-Mixed",
        "bf16": "TensorRT-BF16",
        "fp32": "TensorRT-FP32",
    }
    tensorrt_backend_name = backend_names.get(engine_mode, "TensorRT")
    image_paths = list_images(Path(args.image_dir), args.samples)

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
        str(onnx_path),
        sess_options=ort_options,
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )
    active_providers = ort_session.get_providers()
    if "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "ONNX Runtime CUDA provider failed to initialize. Active providers: "
            + ", ".join(active_providers)
        )
    ort_session.disable_fallback()
    trt_runner = TensorRTRunner(engine_path, trt)

    template = {
        "max_abs": 0.0,
        "abs_sum": 0.0,
        "values": 0,
        "equal": 0,
        "pixels": 0,
        "reference_nonfinite": 0,
        "candidate_nonfinite": 0,
    }
    pytorch_totals = template.copy()
    ort_totals = template.copy()
    first_rgb = None
    first_trt_output = None
    for index, image_path in enumerate(image_paths):
        rgb_numpy = load_rgb_image(image_path, args.input_size)
        rgb_cuda = torch.from_numpy(rgb_numpy).cuda()
        trt_output = trt_runner.make_output(rgb_cuda)
        if first_rgb is None:
            first_rgb = rgb_cuda
            first_trt_output = trt_output
        with torch.inference_mode():
            pytorch_output = wrapper(rgb_cuda).float().cpu().numpy()
            trt_numpy = (
                trt_runner.infer(rgb_cuda, trt_output).float().cpu().numpy()
            )
        ort_output = ort_session.run(None, {"rgb_images": rgb_numpy})[0].astype(
            np.float32
        )
        update_parity(pytorch_output, trt_numpy, pytorch_totals)
        update_parity(ort_output, trt_numpy, ort_totals)
        print(
            f"Validated {index + 1:02d}/{len(image_paths):02d}: {image_path.name}",
            end="\r",
            flush=True,
        )
    print()

    pytorch_parity = parity_summary(pytorch_totals)
    ort_parity = parity_summary(ort_totals)
    print(f"TensorRT: {trt.__version__}")
    print(f"Engine mode: {engine_mode}")
    print(f"Engine: {engine_path.stat().st_size / 2**20:.2f} MiB")
    print(
        f"TensorRT vs PyTorch ({len(image_paths)} images): "
        f"max_abs={pytorch_parity[0]:.6g}, "
        f"finite_mean_abs={pytorch_parity[1]:.6g}, "
        f"argmax_agreement={pytorch_parity[2]:.5f}%, "
        f"reference_nonfinite={pytorch_parity[3]}, "
        f"tensorrt_nonfinite={pytorch_parity[4]}"
    )
    print(
        f"TensorRT vs ONNX Runtime ({len(image_paths)} images): "
        f"max_abs={ort_parity[0]:.6g}, "
        f"finite_mean_abs={ort_parity[1]:.6g}, "
        f"argmax_agreement={ort_parity[2]:.5f}%, "
        f"reference_nonfinite={ort_parity[3]}, "
        f"tensorrt_nonfinite={ort_parity[4]}"
    )

    if any(
        (
            pytorch_parity[3],
            pytorch_parity[4],
            ort_parity[3],
            ort_parity[4],
        )
    ):
        raise RuntimeError(
            "Non-finite output values detected before benchmarking: "
            f"PyTorch={pytorch_parity[3]}, ORT={ort_parity[3]}, "
            f"TensorRT={pytorch_parity[4]}"
        )
    if pytorch_parity[2] < args.min_agreement:
        raise RuntimeError(
            f"TensorRT/PyTorch argmax agreement {pytorch_parity[2]:.5f}% "
            f"is below the {args.min_agreement:.5f}% gate"
        )
    print("Numerical gate: PASSED")

    pytorch_latencies = benchmark_pytorch(wrapper, first_rgb, args.warmup, args.iters)
    ort_latencies = benchmark_onnxruntime(
        ort_session, first_rgb, args.warmup, args.iters
    )
    trt_latencies = benchmark_tensorrt(
        trt_runner, first_rgb, first_trt_output, args.warmup, args.iters
    )

    rows = [
        summarize_latency("PyTorch-FP16", pytorch_latencies),
        summarize_latency("ONNXRuntime-CUDA", ort_latencies),
        summarize_latency(tensorrt_backend_name, trt_latencies),
    ]
    print()
    print("| Backend          | Mean(ms) | P50(ms) | P90(ms) | Img/s |")
    print("|------------------|---------:|--------:|--------:|------:|")
    for name, mean_ms, median_ms, p90_ms, throughput in rows:
        print(
            f"| {name:<16} | {mean_ms:8.3f} | {median_ms:7.3f} | "
            f"{p90_ms:7.3f} | {throughput:5.2f} |"
        )

    trt_mean = rows[2][1]
    print(
        f"\nTensorRT vs PyTorch: {rows[0][1] / trt_mean:.3f}x speedup; "
        f"TensorRT vs ORT: {rows[1][1] / trt_mean:.3f}x speedup"
    )
    print("Deployment gate: PASSED")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
