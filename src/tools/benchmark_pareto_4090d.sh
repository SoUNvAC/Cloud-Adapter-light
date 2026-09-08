#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

precision="${1:-fp16}"
iters="${2:-100}"

run_case() {
  local name="$1"
  local config="$2"
  local work_dir="$3"
  local checkpoints=("${work_dir}"/best_mIoU_iter_*.pth)

  if ((${#checkpoints[@]} == 0)); then
    echo "${name}: skipped (no best checkpoint in ${work_dir})"
    return 0
  fi
  if ((${#checkpoints[@]} > 1)); then
    echo "${name}: expected one best checkpoint in ${work_dir}, found ${#checkpoints[@]}"
    return 1
  fi

  echo "===== ${name} ====="
  python tools/benchmark_deployment.py \
    --config "${config}" \
    --checkpoint "${checkpoints[0]}" \
    --batch-size 1 \
    --precision "${precision}" \
    --iters "${iters}"
}

run_case \
  "V1-LightHead" \
  "configs/light/cloud_adapter_dinov2_s_light_l1c.py" \
  "work_dirs/cloud_adapter_dinov2_s_light_l1c"

run_case \
  "Control-Mask2Former" \
  "configs/control/cloud_adapter_dinov2_s_mask2former_4x_l1c.py" \
  "work_dirs/cloud_adapter_dinov2_s_mask2former_4x_l1c"

run_case \
  "V5-Q50-D3-P3" \
  "configs/light/cloud_adapter_dinov2_s_mask2former_lite_l1c.py" \
  "work_dirs/cloud_adapter_dinov2_s_mask2former_lite_l1c"

run_case \
  "V6-Q25-D3-P3" \
  "configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_l1c.py" \
  "work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_l1c"

run_case \
  "V7-Q25-D2-P3" \
  "configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c.py" \
  "work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c"

run_case \
  "V8-Q25-D2-P2" \
  "configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py" \
  "work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c"

run_case \
  "V9-Micro-Q25-D2-P2" \
  "configs/light/cloud_adapter_dinov2_s_mask2former_micro_q25_d2_p2_l1c.py" \
  "work_dirs/cloud_adapter_dinov2_s_mask2former_micro_q25_d2_p2_l1c"
