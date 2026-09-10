#!/usr/bin/env bash
set -euo pipefail

source tools/activate_onnxruntime_cuda_libs.sh

source_onnx="work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx"
output_dir="work_dirs/phase20_standalone_mask"
mask_onnx="${output_dir}/v12_export_fpn_fp16_mask_512.onnx"
iterations="${1:-200}"

if [[ ! -f "${source_onnx}" ]]; then
  echo "Missing source ONNX: ${source_onnx}"
  exit 1
fi
mkdir -p "${output_dir}"

python tools/prepare_phase20_mask_onnx.py \
  --input "${source_onnx}" \
  --output "${mask_onnx}" \
  --input-size 512 \
  --force

python tools/validate_phase20_standalone_ort.py \
  --mask-onnx "${mask_onnx}" \
  --logits-onnx "${source_onnx}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/test \
  --ann-dir data/cloudsen12_high_l1c/ann_dir/test \
  --input-size 512 \
  --expected-samples 975 \
  --parity-samples 20 \
  --warmup 20 \
  --iters "${iterations}" \
  --expected-miou 68.728 \
  --max-miou-delta 0.05 \
  --min-agreement 99.99
