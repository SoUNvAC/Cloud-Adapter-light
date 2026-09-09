import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a static TensorRT 10 FP16 engine from the Phase 12 ONNX"
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def tensor_description(engine, trt):
    tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
            }
        )
    return tensors


def main():
    args = parse_args()
    if args.workspace_gib <= 0:
        raise ValueError("workspace-gib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "Install the pinned CUDA 12 builder with: "
            "python -m pip install tensorrt-cu12==10.0.1"
        ) from error

    major = int(trt.__version__.split(".", maxsplit=1)[0])
    if major != 10:
        raise RuntimeError(
            f"This reproducible build path requires TensorRT 10.x, found {trt.__version__}"
        )

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Engine already exists: {output_path} (use --force)")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(errors))

    if not builder.platform_has_fast_fp16:
        raise RuntimeError("The current GPU does not report fast native FP16 support")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 2**30)
    )
    config.set_flag(trt.BuilderFlag.FP16)
    # Avoid TF32 changing the few graph regions that intentionally remain FP32.
    config.clear_flag(trt.BuilderFlag.TF32)
    print(
        f"Building TensorRT {trt.__version__} FP16 engine with "
        f"{args.workspace_gib:.1f} GiB workspace..."
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the serialized engine")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("The newly built engine could not be deserialized")

    metadata = {
        "source_onnx": str(onnx_path),
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "workspace_gib": args.workspace_gib,
        "engine_size_mib": output_path.stat().st_size / 2**20,
        "io_tensors": tensor_description(engine, trt),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Engine size: {metadata['engine_size_mib']:.2f} MiB")
    for tensor in metadata["io_tensors"]:
        print(
            f"{tensor['mode']}: {tensor['name']} "
            f"{tensor['dtype']}{tensor['shape']}"
        )
    print(f"Saved engine: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
