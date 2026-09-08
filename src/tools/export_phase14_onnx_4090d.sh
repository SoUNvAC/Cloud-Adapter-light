#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"
checkpoint_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
output_dir="work_dirs/phase14_v12_onnx"
output_path="${output_dir}/v12_export_fpn_fp16_512.onnx"
checkpoints=("${checkpoint_dir}"/best_mIoU_iter_*.pth)

if ((${#checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${checkpoint_dir}, found ${#checkpoints[@]}"
  exit 1
fi

python - <<'PY'
try:
    import onnx
except ImportError as error:
    raise SystemExit(
        "Missing ONNX. Install it with: python -m pip install onnx==1.15.0"
    ) from error
print("onnx:", onnx.__version__)
PY

mkdir -p "${output_dir}"
python tools/export_phase12_onnx.py \
  --config "${config}" \
  --checkpoint "${checkpoints[0]}" \
  --output "${output_path}" \
  --precision fp16 \
  --input-size 512 \
  --opset 17 \
  "$@"
