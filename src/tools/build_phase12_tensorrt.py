import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a static TensorRT 10 engine from the Phase 12 ONNX"
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("mixed-fp16", "fp32"), default="mixed-fp16"
    )
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


def constrain_sensitive_layers(network, trt):
    """Keep overflow-prone reductions and nonlinearities in FP32."""
    sensitive_types = {
        layer_type
        for layer_type in (
            getattr(trt.LayerType, "NORMALIZATION", None),
            getattr(trt.LayerType, "REDUCE", None),
            getattr(trt.LayerType, "SOFTMAX", None),
        )
        if layer_type is not None
    }
    # TensorRT 10.0.1's network.get_layer() exposes a generic ILayer rather
    # than the typed IElementWiseLayer/IUnaryLayer subclasses, so `.op` is not
    # available. ONNX parser layer names retain the operator name and provide
    # a version-stable way to identify these sensitive operations.
    sensitive_name_tokens = (
        "/div",
        "div_",
        "/pow",
        "pow_",
        "/sqrt",
        "sqrt_",
        "/reciprocal",
        "reciprocal_",
    )

    constrained = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        tensors = [
            layer.get_input(tensor_index)
            for tensor_index in range(layer.num_inputs)
        ] + [
            layer.get_output(tensor_index)
            for tensor_index in range(layer.num_outputs)
        ]
        if not any(
            tensor is not None and tensor.dtype in (trt.float16, trt.float32)
            for tensor in tensors
        ):
            continue
        name = layer.name.lower()
        force_fp32 = layer.type in sensitive_types or any(
            token in name
            for token in (
                "layernorm",
                "layer_norm",
                "groupnorm",
                "group_norm",
                "instancenorm",
                "instance_norm",
                "/norm",
            )
        ) or any(token in name for token in sensitive_name_tokens)
        if not force_fp32:
            continue

        layer.precision = trt.float32
        for output_index in range(layer.num_outputs):
            output = layer.get_output(output_index)
            if output is not None and output.dtype in (trt.float16, trt.float32):
                layer.set_output_type(output_index, trt.float32)
        constrained.append(layer.name)
    return constrained


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
    constrained_layers = []
    if args.mode == "mixed-fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        constrained_layers = constrain_sensitive_layers(network, trt)
        if not constrained_layers:
            raise RuntimeError("No numerically sensitive layers were constrained")
    else:
        # Strict FP32 is a diagnostic fallback, not the desired final engine.
        config.clear_flag(trt.BuilderFlag.TF32)
    print(
        f"Building TensorRT {trt.__version__} {args.mode} engine with "
        f"{args.workspace_gib:.1f} GiB workspace..."
    )
    if constrained_layers:
        print(f"FP32-constrained sensitive layers: {len(constrained_layers)}")
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
        "mode": args.mode,
        "workspace_gib": args.workspace_gib,
        "fp32_constrained_layer_count": len(constrained_layers),
        "fp32_constrained_layers": constrained_layers,
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
