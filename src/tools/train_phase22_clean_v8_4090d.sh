#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seed="${1:-42}"
config="configs/protocol/phase22_clean_v8_l1c.py"
work_dir="work_dirs/phase22_clean_v8/seed${seed}"
resume_args=()

if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f \
    -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/val_eval.log" ]]; then
    echo "Reusing completed Phase 22 run: ${work_dir}"
    exit 0
  fi
  if [[ -f "${work_dir}/last_checkpoint" ]]; then
    echo "Resuming interrupted Phase 22 run: ${work_dir}"
    resume_args=(--resume)
  else
    echo "Refusing incomplete run without last_checkpoint: ${work_dir}" >&2
    exit 2
  fi
fi

if [[ -d "data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="data/cloudsen12_high_l1c"
elif [[ -d "../data/cloudsen12_high_l1c" ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="../data/cloudsen12_high_l1c"
else
  echo "CloudSEN12 High L1C is missing from src/data and repository data" >&2
  exit 6
fi

bash tools/train_light_4090d.sh \
  "${config}" \
  "${work_dir}" \
  "${resume_args[@]}" \
  --cfg-options \
  "randomness.seed=${seed}" \
  "randomness.deterministic=True"

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f \
  -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one best checkpoint, found ${#checkpoints[@]}" >&2
  exit 3
fi

# Deliberately redirect the test loop to val. Phase 22 must not observe test.
python tools/test.py \
  "${config}" \
  "${checkpoints[0]}" \
  --work-dir "${work_dir}/val_eval" \
  --cfg-options \
  "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
  "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
  "randomness.seed=${seed}" \
  "randomness.deterministic=True" \
  2>&1 | tee "${work_dir}/val_eval.log"
