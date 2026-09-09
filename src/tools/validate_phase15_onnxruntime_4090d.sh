#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

python - <<'PY'
import importlib.util
import sys

if not ((3, 8) <= sys.version_info[:2] <= (3, 11)):
    version = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(
        f"Unsupported Python {version}. Activate the Cloud-Adapter Python 3.10 "
        "training environment before Phase 15."
    )

required = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime-gpu (CUDA 12 build)",
    "torch": "torch",
}
missing = [
    package
    for module, package in required.items()
    if importlib.util.find_spec(module) is None
]
if missing:
    raise SystemExit(
        "Missing Phase 15 dependencies: "
        + ", ".join(missing)
        + ". Run: bash tools/install_onnxruntime_cu12.sh"
    )

import torch
import onnxruntime as ort

print(f"Python: {sys.version.split()[0]} | executable: {sys.executable}")
print(
    f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | "
    f"ONNX Runtime: {ort.__version__}"
)
PY

config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"
checkpoint_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
onnx_path="work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx"
checkpoints=("${checkpoint_dir}"/best_mIoU_iter_*.pth)

if ((${#checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${checkpoint_dir}, found ${#checkpoints[@]}"
  exit 1
fi
if [[ ! -f "${onnx_path}" ]]; then
  echo "Missing ${onnx_path}; run tools/export_phase14_onnx_4090d.sh first"
  exit 1
fi

python tools/validate_phase12_onnxruntime.py \
  --config "${config}" \
  --checkpoint "${checkpoints[0]}" \
  --onnx "${onnx_path}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/test \
  --samples 20 \
  --warmup 20 \
  --iters 100 \
  --input-size 512 \
  "$@"
