#!/usr/bin/env bash

# This file must be sourced so LD_LIBRARY_PATH reaches the Python process that
# loads libonnxruntime_providers_cuda.so.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead of executing it:"
  echo "  source tools/activate_onnxruntime_cuda_libs.sh"
  exit 1
fi

ort_cuda_library_path="$(python - <<'PY'
from pathlib import Path
import site
import sys

import torch

candidates = []


def add(path):
    path = Path(path)
    if path.is_dir() and path not in candidates:
        candidates.append(path)


add(Path(torch.__file__).resolve().parent / "lib")
for site_dir in site.getsitepackages():
    nvidia_dir = Path(site_dir) / "nvidia"
    if nvidia_dir.is_dir():
        for library_dir in sorted(nvidia_dir.glob("*/lib")):
            add(library_dir)
add(Path(sys.prefix) / "lib")
add("/usr/local/cuda/lib64")

print(":".join(str(path) for path in candidates))
PY
)"

if [[ -z "${ort_cuda_library_path}" ]]; then
  echo "Could not locate CUDA runtime libraries in the active Python environment."
  return 1
fi

export LD_LIBRARY_PATH="${ort_cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset ort_cuda_library_path
