#!/usr/bin/env bash
set -euo pipefail

# Cloud-Adapter's pinned torch 2.1.0 stack uses CUDA 12.1 and cuDNN 8.
# ORT 1.18.0 from normal PyPI is the CUDA 11.8 build, so install the
# CUDA 12/cuDNN 8 wheel from Microsoft's official CUDA 12 feed instead.
python - <<'PY'
import sys

if not ((3, 8) <= sys.version_info[:2] <= (3, 11)):
    version = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(
        f"Unsupported Python {version}. Activate the Cloud-Adapter training "
        "environment (Python 3.10 recommended) before running this script."
    )

try:
    import torch
except ImportError as error:
    raise SystemExit(
        "PyTorch is missing. Activate the Cloud-Adapter training environment "
        "before installing ONNX Runtime."
    ) from error

cuda = torch.version.cuda
cudnn = torch.backends.cudnn.version()
if cuda is None or not cuda.startswith("12."):
    raise SystemExit(
        f"Expected the CUDA 12 PyTorch build, but torch reports CUDA {cuda}."
    )
if cudnn is None or cudnn // 1000 != 8:
    raise SystemExit(
        f"Expected the cuDNN 8 PyTorch build, but torch reports cuDNN {cudnn}."
    )

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} | CUDA: {cuda} | cuDNN: {cudnn}")
PY

# Install ordinary dependencies from PyPI first. --only-binary prevents a
# wrong/new Python interpreter from spending time compiling NumPy from source.
python -m pip install --only-binary=:all: \
  "numpy==1.26.4" \
  "opencv-python-headless==4.10.0.84" \
  "onnx==1.15.0" \
  coloredlogs flatbuffers packaging protobuf sympy

# Remove either CPU ORT or the CUDA 11 wheel so the Python module cannot be
# shadowed by a second package with the same import name.
python -m pip uninstall -y onnxruntime onnxruntime-gpu || true
python -m pip install \
  --only-binary=:all: \
  --no-cache-dir \
  --no-deps \
  --index-url \
  https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  "onnxruntime-gpu==1.18.0"

# PyTorch's pip wheels keep CUDA libraries below site-packages/nvidia/*/lib.
# Make those directories visible to ORT's provider dlopen call on Linux.
source tools/activate_onnxruntime_cuda_libs.sh

ort_cuda_provider_lib="$(python - <<'PY'
from pathlib import Path
import onnxruntime as ort

print(Path(ort.__file__).resolve().parent / "capi/libonnxruntime_providers_cuda.so")
PY
)"
if [[ ! -f "${ort_cuda_provider_lib}" ]]; then
  echo "Missing ONNX Runtime CUDA provider library: ${ort_cuda_provider_lib}"
  exit 1
fi
ort_missing_libraries="$(ldd "${ort_cuda_provider_lib}" | awk '/not found/')"
if [[ -n "${ort_missing_libraries}" ]]; then
  echo "ONNX Runtime CUDA provider still has unresolved shared libraries:"
  echo "${ort_missing_libraries}"
  echo "Resolved search path: ${LD_LIBRARY_PATH}"
  exit 1
fi
unset ort_cuda_provider_lib ort_missing_libraries

# Import torch first so its matching CUDA/cuDNN shared libraries are loaded
# before ONNX Runtime tries to initialize its CUDA execution provider.
python - <<'PY'
import torch
import cv2
import numpy
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(
        "CUDAExecutionProvider is not present after installation: "
        + ", ".join(providers)
    )

# get_available_providers() only describes the wheel build. Creating a session
# actually dlopens libonnxruntime_providers_cuda.so and all CUDA dependencies.
input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
graph = helper.make_graph(
    [helper.make_node("Add", ["input", "one"], ["output"])],
    "cuda_provider_smoke_test",
    [input_info],
    [output_info],
    [one],
)
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 17)],
)
session = ort.InferenceSession(
    model.SerializeToString(),
    providers=[("CUDAExecutionProvider", {"device_id": 0})],
)
active_providers = session.get_providers()
if "CUDAExecutionProvider" not in active_providers:
    raise SystemExit(
        "CUDAExecutionProvider is present in the wheel but failed to initialize. "
        "Active providers: " + ", ".join(active_providers)
    )
session.disable_fallback()
actual = session.run(None, {"input": numpy.array([2.0], dtype=numpy.float32)})[0]
if not numpy.array_equal(actual, numpy.array([3.0], dtype=numpy.float32)):
    raise SystemExit(f"Unexpected CUDA smoke-test output: {actual}")

print(f"NumPy: {numpy.__version__} | OpenCV: {cv2.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print("Active session providers: " + ", ".join(active_providers))
print("ONNX Runtime CUDA 12 installation: OK")
PY
