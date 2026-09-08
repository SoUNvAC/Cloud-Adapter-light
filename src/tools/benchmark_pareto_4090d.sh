#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

precision="${1:-fp16}"
iters="${2:-100}"
benchmark_gpu=""
rows=()

run_case() {
  local name="$1"
  local config="$2"
  local work_dir="$3"
  local checkpoints=("${work_dir}"/best_mIoU_iter_*.pth)

  if ((${#checkpoints[@]} == 0)); then
    printf 'Skipping %s: no best checkpoint found\n' "${name}" >&2
    rows+=("${name}"$'\t-'$'\t-'$'\t-'$'\t-'$'\t-'$'\t-'$'\t-')
    return 0
  fi
  if ((${#checkpoints[@]} > 1)); then
    echo "${name}: expected one best checkpoint in ${work_dir}, found ${#checkpoints[@]}"
    return 1
  fi

  printf 'Benchmarking %s...\n' "${name}" >&2
  local metrics
  metrics="$(python tools/benchmark_deployment.py \
    --config "${config}" \
    --checkpoint "${checkpoints[0]}" \
    --batch-size 1 \
    --precision "${precision}" \
    --iters "${iters}" \
    --quiet \
    --output-format tsv)"

  local gpu params checkpoint mean median p90 throughput peak
  IFS=$'\t' read -r gpu params checkpoint mean median p90 throughput peak <<< "${metrics}"
  if [[ -z "${benchmark_gpu}" ]]; then
    benchmark_gpu="${gpu}"
  fi
  rows+=("${name}"$'\t'"${params}"$'\t'"${checkpoint}"$'\t'"${mean}"$'\t'"${median}"$'\t'"${p90}"$'\t'"${throughput}"$'\t'"${peak}")
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

printf '\nGPU: %s | Precision: %s | Batch: 1 | Input: 512x512 | Iterations: %s\n\n' \
  "${benchmark_gpu:-unknown}" "${precision}" "${iters}"
printf '| %-25s | %10s | %10s | %9s | %9s | %9s | %9s | %10s |\n' \
  "Model" "Params(M)" "Ckpt(MiB)" "Mean(ms)" "P50(ms)" "P90(ms)" "Img/s" "Peak(GiB)"
printf '|---------------------------|-----------:|-----------:|----------:|----------:|----------:|----------:|-----------:|\n'

for row in "${rows[@]}"; do
  IFS=$'\t' read -r name params checkpoint mean median p90 throughput peak <<< "${row}"
  printf '| %-25s | %10s | %10s | %9s | %9s | %9s | %9s | %10s |\n' \
    "${name}" "${params}" "${checkpoint}" "${mean}" "${median}" "${p90}" "${throughput}" "${peak}"
done
