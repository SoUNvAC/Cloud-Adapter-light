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

# TensorRT <=10.8 marks the builder resource as requiring an executable stack.
# New glibc releases reject that during dlopen. NVIDIA confirmed this flag is
# unnecessary; clear it only on the affected TensorRT library. Reinstalling the
# tensorrt-cu12-libs wheel restores the vendor copy if that is ever required.
python -m pip install --only-binary=:all: "patchelf==0.19.1.0"
trt_builder_resource="$(python - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

try:
    package = distribution("tensorrt-cu12-libs")
except PackageNotFoundError as error:
    raise SystemExit("The tensorrt-cu12-libs package is missing") from error

matches = [
    package.locate_file(entry).resolve()
    for entry in (package.files or [])
    if Path(entry).name == "libnvinfer_builder_resource.so.10.0.1"
]
if len(matches) != 1:
    raise SystemExit(
        "Expected one libnvinfer_builder_resource.so.10.0.1, found: "
        + ", ".join(map(str, matches))
    )
print(matches[0])
PY
)"
patchelf_bin="$(python - <<'PY'
import shutil

path = shutil.which("patchelf")
if path is None:
    raise SystemExit("patchelf executable was not installed")
print(path)
PY
)"

stack_before="$("${patchelf_bin}" --print-execstack "${trt_builder_resource}")"
"${patchelf_bin}" --clear-execstack "${trt_builder_resource}"
stack_after="$("${patchelf_bin}" --print-execstack "${trt_builder_resource}")"
if [[ "${stack_after}" == *X* ]]; then
  echo "Failed to clear executable-stack flag: ${stack_after}"
  exit 1
fi
echo "TensorRT builder GNU_STACK: ${stack_before} -> ${stack_after}"
unset trt_builder_resource patchelf_bin stack_before stack_after

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
