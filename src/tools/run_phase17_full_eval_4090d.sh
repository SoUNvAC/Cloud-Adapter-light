#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

source tools/activate_onnxruntime_cuda_libs.sh

config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"
checkpoint_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
onnx_path="work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx"
engine_path="work_dirs/phase16_v12_tensorrt/v12_export_fpn_fp32_b1_512.engine"
checkpoints=("${checkpoint_dir}"/best_mIoU_iter_*.pth)

if ((${#checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${checkpoint_dir}, found ${#checkpoints[@]}"
  exit 1
fi
for artifact in "${onnx_path}" "${engine_path}"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "Missing deployment artifact: ${artifact}"
    exit 1
  fi
done

python tools/eval_phase17_full_deployment.py \
  --config "${config}" \
  --checkpoint "${checkpoints[0]}" \
  --onnx "${onnx_path}" \
  --engine "${engine_path}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/test \
  --ann-dir data/cloudsen12_high_l1c/ann_dir/test \
  --input-size 512 \
  --expected-samples 975 \
  --expected-pytorch-miou 68.73 \
  --max-reference-miou-delta 0.05 \
  --min-agreement 99.9 \
  --max-miou-delta 0.05 \
  "$@"
