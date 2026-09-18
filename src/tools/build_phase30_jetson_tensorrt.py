import argparse
import hashlib
import json
import platform
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a target-local TensorRT 10 mask engine without PyTorch."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("mixed-fp16", "int8-qdq"), required=True)
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def device_model():
    path = Path("/proc/device-tree/model")
    if path.is_file():
        return path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
    return platform.platform()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_description(engine, trt):
    return [
        {
            "name": engine.get_tensor_name(index),
            "mode": str(engine.get_tensor_mode(engine.get_tensor_name(index))),
            "dtype": str(engine.get_tensor_dtype(engine.get_tensor_name(index))),
            "shape": list(engine.get_tensor_shape(engine.get_tensor_name(index))),
        }
        for index in range(engine.num_io_tensors)
    ]


def constrain_sensitive_layers(network, trt):
    sensitive_types = {
        value
        for value in (
            getattr(trt.LayerType, "NORMALIZATION", None),
            getattr(trt.LayerType, "REDUCE", None),
            getattr(trt.LayerType, "SOFTMAX", None),
        )
        if value is not None
    }
    name_tokens = (
        "layernorm",
        "layer_norm",
        "groupnorm",
        "group_norm",
        "instancenorm",
        "instance_norm",
        "/norm",
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
        if layer.type not in sensitive_types and not any(
            token in layer.name.lower() for token in name_tokens
        ):
            continue
        layer.precision = trt.float32
        for output_index in range(layer.num_outputs):
            output = layer.get_output(output_index)
            if output is not None and output.dtype in (trt.float16, trt.float32):
                layer.set_output_type(output_index, trt.float32)
        constrained.append(layer.name)
    return constrained


def count_qdq(path):
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx==1.15.0 is required") from error
    model = onnx.load(str(path), load_external_data=False)
    counts = {"QuantizeLinear": 0, "DequantizeLinear": 0}
    for node in model.graph.node:
        if node.op_type in counts:
            counts[node.op_type] += 1
    return counts


def main():
    args = parse_args()
    if args.workspace_gib <= 0:
        raise ValueError("workspace-gib must be positive")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python bindings are required on the Jetson") from error
    if int(trt.__version__.split(".", maxsplit=1)[0]) != 10:
        raise RuntimeError(f"Phase 30 requires TensorRT 10.x, found {trt.__version__}")
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output exists: {output_path} (use --force)")
    qdq_counts = count_qdq(onnx_path)
    if args.mode == "int8-qdq" and not all(qdq_counts.values()):
        raise RuntimeError("INT8 mode requires explicit Q/DQ nodes")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 2**30)
    )
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("Target does not report fast native FP16")
    config.set_flag(trt.BuilderFlag.FP16)
    if args.mode == "int8-qdq":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Target does not report fast native INT8")
        config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    constrained = constrain_sensitive_layers(network, trt)
    if not constrained:
        raise RuntimeError("No numerically sensitive layers were constrained")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the target engine")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Built engine could not be deserialized")
    tensors = tensor_description(engine, trt)
    inputs = [row for row in tensors if "INPUT" in row["mode"]]
    outputs = [row for row in tensors if "OUTPUT" in row["mode"]]
    if len(inputs) != 1 or inputs[0]["name"] != "rgb_images":
        raise RuntimeError(f"Unexpected engine inputs: {inputs}")
    if len(outputs) != 1 or outputs[0]["name"] != "seg_mask":
        raise RuntimeError(f"Unexpected engine outputs: {outputs}")
    metadata = {
        "phase": 30,
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": sha256_file(onnx_path),
        "mode": args.mode,
        "tensorrt_version": trt.__version__,
        "device_model": device_model(),
        "machine": platform.machine(),
        "workspace_gib": args.workspace_gib,
        "explicit_qdq_counts": qdq_counts,
        "fp32_constrained_layer_count": len(constrained),
        "fp32_constrained_layers": constrained,
        "engine_size_mib": output_path.stat().st_size / 2**20,
        "engine_sha256": sha256_file(output_path),
        "io_tensors": tensors,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
