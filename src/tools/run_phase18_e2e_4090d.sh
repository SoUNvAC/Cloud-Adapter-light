#!/usr/bin/env bash
set -euo pipefail

source tools/activate_onnxruntime_cuda_libs.sh

onnx_path="work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx"
engine_path="work_dirs/phase16_v12_tensorrt/v12_export_fpn_fp32_b1_512.engine"
iterations="${1:-200}"

for artifact in "${onnx_path}" "${engine_path}" "${engine_path}.json"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "Missing deployment artifact: ${artifact}"
    exit 1
  fi
done

python tools/benchmark_phase18_e2e.py \
  --onnx "${onnx_path}" \
  --engine "${engine_path}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/test \
  --samples 20 \
  --warmup 20 \
  --iters "${iterations}" \
  --input-size 512
