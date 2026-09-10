#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

source tools/activate_onnxruntime_cuda_libs.sh

config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"
checkpoint_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
output_dir="work_dirs/phase19_resolution_pareto"
baseline_onnx="work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx"
iterations="${1:-200}"
checkpoints=("${checkpoint_dir}"/best_mIoU_iter_*.pth)

if ((${#checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${checkpoint_dir}, found ${#checkpoints[@]}"
  exit 1
fi
if [[ ! -f "${baseline_onnx}" ]]; then
  echo "Missing baseline ONNX: ${baseline_onnx}"
  exit 1
fi

mkdir -p "${output_dir}"
model_args=(--model "512=${baseline_onnx}")
for size in 448 320; do
  onnx_path="${output_dir}/v12_export_fpn_fp16_${size}.onnx"
  python tools/export_phase12_onnx.py \
    --config "${config}" \
    --checkpoint "${checkpoints[0]}" \
    --output "${onnx_path}" \
    --precision fp16 \
    --input-size "${size}" \
    --opset 17 \
    --force
  model_args+=(--model "${size}=${onnx_path}")
done

python tools/eval_phase19_resolution_pareto.py \
  "${model_args[@]}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/test \
  --ann-dir data/cloudsen12_high_l1c/ann_dir/test \
  --expected-samples 975 \
  --benchmark-samples 20 \
  --warmup 20 \
  --iters "${iterations}" \
  --output-size 512 \
  --baseline-size 512 \
  --expected-baseline-miou 68.728 \
  --max-baseline-miou-delta 0.05 \
  --max-miou-drop 0.50 \
  --min-speedup 1.15
