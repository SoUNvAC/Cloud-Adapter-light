#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

config="configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py"
source_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c"
artifact_dir="work_dirs/phase11_v8_quant_screen"
iters="${1:-100}"
source_checkpoints=("${source_dir}"/best_mIoU_iter_*.pth)

if ((${#source_checkpoints[@]} != 1)); then
  echo "Expected one V8 best checkpoint in ${source_dir}, found ${#source_checkpoints[@]}"
  exit 1
fi
if [[ ! -f "${artifact_dir}/v8_fp16.pth" ]]; then
  echo "Missing ${artifact_dir}/v8_fp16.pth; run the Phase 11 preparation first"
  exit 1
fi

benchmark_gpu=""
rows=()

run_case() {
  local name="$1"
  local checkpoint="$2"
  local weight_dtype="$3"

  printf 'Benchmarking %s...\n' "${name}" >&2
  local metrics
  metrics="$(python tools/benchmark_deployment.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --batch-size 1 \
    --precision fp16 \
    --weight-dtype "${weight_dtype}" \
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

run_case "V8-Autocast" "${source_checkpoints[0]}" "fp32"
run_case "V8-Native-FP16" "${artifact_dir}/v8_fp16.pth" "fp16"

printf '\nGPU: %s | Compute: fp16 | Batch: 1 | Input: 512x512 | Iterations: %s\n\n' \
  "${benchmark_gpu:-unknown}" "${iters}"
printf '| %-16s | %10s | %10s | %9s | %9s | %9s | %9s | %10s |\n' \
  "Artifact" "Params(M)" "Ckpt(MiB)" "Mean(ms)" "P50(ms)" "P90(ms)" "Img/s" "Peak(GiB)"
printf '|------------------|-----------:|-----------:|----------:|----------:|----------:|----------:|-----------:|\n'
for row in "${rows[@]}"; do
  IFS=$'\t' read -r name params checkpoint_size mean median p90 throughput peak <<< "${row}"
  printf '| %-16s | %10s | %10s | %9s | %9s | %9s | %9s | %10s |\n' \
    "${name}" "${params}" "${checkpoint_size}" "${mean}" "${median}" \
    "${p90}" "${throughput}" "${peak}"
done
