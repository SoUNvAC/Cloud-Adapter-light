#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

config="configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py"
source_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c"
artifact_dir="work_dirs/phase11_v8_quant_screen"
log_dir="${artifact_dir}/eval_logs"
source_checkpoints=("${source_dir}"/best_mIoU_iter_*.pth)

if ((${#source_checkpoints[@]} != 1)); then
  echo "Expected one V8 best checkpoint in ${source_dir}, found ${#source_checkpoints[@]}"
  exit 1
fi

mkdir -p "${log_dir}"
rows=()

evaluate_one() {
  local name="$1"
  local checkpoint="$2"
  local slug="$3"
  local log_path="${log_dir}/${slug}.log"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}"
    exit 1
  fi

  printf 'Evaluating %s...\n' "${name}" >&2
  if ! python tools/test.py \
      "${config}" \
      "${checkpoint}" \
      --work-dir "${artifact_dir}/${slug}" \
      >"${log_path}" 2>&1; then
    echo "Evaluation failed for ${name}; final log lines:" >&2
    tail -80 "${log_path}" >&2
    exit 1
  fi

  local metrics
  metrics="$(python tools/extract_mmseg_metrics.py \
    --log "${log_path}" \
    --checkpoint "${checkpoint}" \
    --output-format tsv)"
  rows+=("${name}"$'\t'"${metrics}")
}

evaluate_one "V8-Original" "${source_checkpoints[0]}" "v8_original"
evaluate_one "V8-FP16-Weights" "${artifact_dir}/v8_fp16.pth" "v8_fp16"
evaluate_one "V8-W8-VFM-Sim" \
  "${artifact_dir}/v8_w8_vfm_sim_fp16.pth" "v8_w8_vfm_sim"
evaluate_one "V8-W8-All-Sim" \
  "${artifact_dir}/v8_w8_all_sim_fp16.pth" "v8_w8_all_sim"

printf '\n| %-19s | %9s | %6s | %6s | %6s | %6s | %7s | %10s | %7s |\n' \
  "Artifact" "Size(MiB)" "aAcc" "mIoU" "mAcc" "mDice" "mFscore" "mPrecision" "mRecall"
printf '|---------------------|----------:|-------:|-------:|-------:|-------:|--------:|-----------:|--------:|\n'
for row in "${rows[@]}"; do
  IFS=$'\t' read -r name size aacc miou macc mdice mfscore mprecision mrecall <<< "${row}"
  printf '| %-19s | %9s | %6s | %6s | %6s | %6s | %7s | %10s | %7s |\n' \
    "${name}" "${size}" "${aacc}" "${miou}" "${macc}" "${mdice}" \
    "${mfscore}" "${mprecision}" "${mrecall}"
done

printf '\nFull evaluation logs: %s\n' "${log_dir}"
