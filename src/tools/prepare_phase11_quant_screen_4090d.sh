#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

source_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c"
output_dir="work_dirs/phase11_v8_quant_screen"
source_checkpoints=("${source_dir}"/best_mIoU_iter_*.pth)

if ((${#source_checkpoints[@]} != 1)); then
  echo "Expected one V8 best checkpoint in ${source_dir}, found ${#source_checkpoints[@]}"
  exit 1
fi

mkdir -p "${output_dir}"
source_checkpoint="${source_checkpoints[0]}"

export_one() {
  local output="$1"
  shift
  if [[ -f "${output}" ]]; then
    echo "Reusing ${output}"
    return 0
  fi
  python tools/export_quant_screen_checkpoint.py \
    --input "${source_checkpoint}" \
    --output "${output}" \
    "$@"
}

export_one "${output_dir}/v8_fp16.pth" \
  --mode fp16

export_one "${output_dir}/v8_w8_vfm_sim_fp16.pth" \
  --mode w8-sim \
  --scope vfm

export_one "${output_dir}/v8_w8_all_sim_fp16.pth" \
  --mode w8-sim \
  --scope all

echo
echo "Phase 11 quantization-screen checkpoints are ready in ${output_dir}"
