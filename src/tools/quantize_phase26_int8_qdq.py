import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a calibrated explicit-Q/DQ INT8 ONNX for Phase 26."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-image-dir", required=True)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def list_images(root: Path):
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No calibration images found in {root}")
    return paths


def select_evenly_spaced(paths, count):
    if count <= 0:
        raise ValueError("calibration-samples must be positive")
    if count >= len(paths):
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("Deterministic calibration sampler produced duplicates")
    return [paths[int(index)] for index in indices]


def load_rgb(path: Path, size: int):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def calibration_manifest_digest(root: Path, paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("input-size must be positive")
    source = Path(args.onnx)
    output = Path(args.output)
    calibration_root = Path(args.calibration_image_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output} (use --force)")

    try:
        import onnx
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install onnx==1.15.0 and onnxruntime-gpu==1.18.0 before Phase 26"
        ) from error

    model = onnx.load(str(source), load_external_data=False)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "rgb_images":
        raise RuntimeError("Expected exactly one ONNX input named rgb_images")

    all_paths = list_images(calibration_root)
    selected = select_evenly_spaced(all_paths, args.calibration_samples)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._iterator = iter(selected)

        def get_next(self):
            try:
                path = next(self._iterator)
            except StopIteration:
                return None
            return {"rgb_images": load_rgb(path, args.input_size)}

        def rewind(self):
            self._iterator = iter(selected)

    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(source),
        model_output=str(output),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "DedicatedQDQPair": True,
        },
    )
    quantized = onnx.load(str(output), load_external_data=False)
    onnx.checker.check_model(quantized)
    op_counts = {}
    for node in quantized.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    if not op_counts.get("QuantizeLinear") or not op_counts.get("DequantizeLinear"):
        raise RuntimeError("Quantized model contains no explicit Q/DQ nodes")

    metadata = {
        "method": "static explicit Q/DQ PTQ",
        "source_onnx": str(source),
        "calibration_split": "val",
        "calibration_image_dir": str(calibration_root),
        "available_calibration_images": len(all_paths),
        "calibration_samples": len(selected),
        "calibration_sampling": "sorted paths, evenly spaced including endpoints",
        "calibration_manifest_sha256": calibration_manifest_digest(
            calibration_root, selected
        ),
        "calibration_method": "MinMax",
        "activation_type": "QInt8 symmetric",
        "weight_type": "QInt8 symmetric per-channel",
        "quantized_op_types": ["Conv", "Gemm", "MatMul"],
        "op_counts": op_counts,
        "source_size_mib": source.stat().st_size / 2**20,
        "quantized_size_mib": output.stat().st_size / 2**20,
        "test_evaluated": False,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
