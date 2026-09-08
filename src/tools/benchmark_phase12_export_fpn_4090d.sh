#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

v8_config="configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py"
v12_config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"
v8_artifact="work_dirs/phase11_v8_quant_screen/v8_fp16.pth"
v12_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
v12_artifact="${v12_dir}/v12_export_fpn_fp16.pth"
iters="${1:-100}"
v12_checkpoints=("${v12_dir}"/best_mIoU_iter_*.pth)

if [[ ! -f "${v8_artifact}" ]]; then
  echo "Missing ${v8_artifact}; run tools/prepare_phase11_quant_screen_4090d.sh first"
  exit 1
fi
if ((${#v12_checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${v12_dir}, found ${#v12_checkpoints[@]}"
  exit 1
fi

if [[ ! -f "${v12_artifact}" ]]; then
  python tools/export_quant_screen_checkpoint.py \
    --input "${v12_checkpoints[0]}" \
    --output "${v12_artifact}" \
    --mode fp16
fi

benchmark_gpu=""
rows=()

run_case() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"

  printf 'Benchmarking %s...\n' "${name}" >&2
  local metrics
  metrics="$(python tools/benchmark_deployment.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --batch-size 1 \
    --precision fp16 \
    --weight-dtype fp16 \
    --iters "${iters}" \
    --quiet \
    --output-format tsv)"

  local gpu params checkpoint_size mean median p90 throughput peak
  IFS=$'\t' read -r gpu params checkpoint_size mean median p90 throughput peak <<< "${metrics}"
  if [[ -z "${benchmark_gpu}" ]]; then
    benchmark_gpu="${gpu}"
  fi
  rows+=("${name}"$'\t'"${params}"$'\t'"${checkpoint_size}"$'\t'"${mean}"$'\t'"${median}"$'\t'"${p90}"$'\t'"${throughput}"$'\t'"${peak}")
}

run_case "V8-Native-FP16" "${v8_config}" "${v8_artifact}"
run_case "V12-ExportFPN-FP16" "${v12_config}" "${v12_artifact}"

printf '\nGPU: %s | Compute/weights: fp16 | Batch: 1 | Input: 512x512 | Iterations: %s\n\n' \
  "${benchmark_gpu:-unknown}" "${iters}"
printf '| %-20s | %10s | %10s | %9s | %9s | %9s | %9s | %10s |\n' \
  "Artifact" "Params(M)" "Ckpt(MiB)" "Mean(ms)" "P50(ms)" "P90(ms)" "Img/s" "Peak(GiB)"
printf '|----------------------|-----------:|-----------:|----------:|----------:|----------:|----------:|-----------:|\n'
for row in "${rows[@]}"; do
  IFS=$'\t' read -r name params checkpoint_size mean median p90 throughput peak <<< "${row}"
  printf '| %-20s | %10s | %10s | %9s | %9s | %9s | %9s | %10s |\n' \
    "${name}" "${params}" "${checkpoint_size}" "${mean}" "${median}" \
    "${p90}" "${throughput}" "${peak}"
done
