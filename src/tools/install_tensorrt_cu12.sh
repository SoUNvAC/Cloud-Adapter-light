#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys

if not ((3, 8) <= sys.version_info[:2] <= (3, 11)):
    version = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(
        f"Unsupported Python {version}. Activate the Cloud-Adapter Python 3.10 "
        "environment before installing TensorRT."
    )

try:
    import torch
except ImportError as error:
    raise SystemExit("PyTorch is missing from the active environment.") from error

cuda = torch.version.cuda
if cuda is None or not cuda.startswith("12."):
    raise SystemExit(f"Expected the CUDA 12 PyTorch build, found CUDA {cuda}.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the NVIDIA GPU.")

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} | CUDA: {cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# TensorRT 10.0.1 is the pinned release tested with CUDA 12.1 and ONNX 1.15.
# The explicit -cu12 package name avoids accidentally resolving a CUDA 13 wheel.
python -m pip install --no-cache-dir "tensorrt-cu12==10.0.1"

source tools/activate_onnxruntime_cuda_libs.sh

python - <<'PY'
import torch
import tensorrt as trt

if not trt.__version__.startswith("10."):
    raise SystemExit(f"Expected TensorRT 10.x, found {trt.__version__}")
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
if builder is None:
    raise SystemExit("TensorRT could not create a CUDA builder")
print(f"TensorRT: {trt.__version__}")
print(f"Fast FP16: {builder.platform_has_fast_fp16}")
print("TensorRT CUDA 12 installation: OK")
PY
