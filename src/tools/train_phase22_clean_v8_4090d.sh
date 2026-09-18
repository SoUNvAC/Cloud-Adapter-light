#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seed="${1:-42}"
config="configs/protocol/phase22_clean_v8_l1c.py"
work_dir="work_dirs/phase22_clean_v8/seed${seed}"

if [[ -e "${work_dir}" ]]; then
  echo "Refusing to reuse existing run directory: ${work_dir}" >&2
  exit 2
fi

bash tools/train_light_4090d.sh \
  "${config}" \
  "${work_dir}" \
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
