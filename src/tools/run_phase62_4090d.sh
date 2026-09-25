#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

root="work_dirs/phase62_partial_labels"
mkdir -p "${root}"

python tools/audit_phase62_partial_labels.py

groups=(complete12 ignore65 partial65)
configs=(
  configs/protocol/phase62_complete12_l1c.py
  configs/protocol/phase62_ignore65_l1c.py
  configs/protocol/phase62_partial65_l1c.py
)
seeds=(62 63 64)

for index in "${!groups[@]}"; do
  group="${groups[$index]}"
  config="${configs[$index]}"
  for seed in "${seeds[@]}"; do
    work_dir="${root}/${group}/seed${seed}"
    mkdir -p "${work_dir}"
    if [[ ! -f "${work_dir}/TRAIN_COMPLETE" ]]; then
      resume_args=()
      if [[ -f "${work_dir}/last_checkpoint" ]]; then
        resume_args=(--resume)
      fi
      PYTHONHASHSEED="${seed}" bash tools/train_light_4090d.sh \
        "${config}" "${work_dir}" "${resume_args[@]}" \
        --cfg-options randomness.seed="${seed}" randomness.deterministic=True
      touch "${work_dir}/TRAIN_COMPLETE"
    fi
  done
done

python -u tools/evaluate_phase62_partial_labels.py \
  2>&1 | tee "${root}/evaluation_console.log"

python -m json.tool "${root}/summary.json"
sha256sum "${root}/preflight.json" "${root}/summary.json" \
  > "${root}/SHA256SUMS.txt"
touch "${root}/PHASE62_COMPLETE"

