#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

candidate="${1:?candidate name required}"
subset="${2:?comma-separated active block subset required}"
seed="${3:?seed required}"
config="configs/protocol/phase25_block_kd_l1c.py"
work_dir="work_dirs/phase25_block_kd/${candidate}/seed${seed}"

if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f \
    -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/val_eval.log" ]]; then
    echo "Reusing completed Phase 25 run: ${work_dir}"
    exit 0
  fi
  echo "Refusing incomplete existing run directory: ${work_dir}" >&2
  exit 2
fi

mapfile -t student_checkpoints < <(find \
  "work_dirs/phase23_clean_v12/seed${seed}" -maxdepth 1 -type f \
  -name 'best_mIoU_iter_*.pth' -print)
mapfile -t teacher_checkpoints < <(find \
  "work_dirs/phase22_clean_v8/seed${seed}" -maxdepth 1 -type f \
  -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#student_checkpoints[@]}" -ne 1 || "${#teacher_checkpoints[@]}" -ne 1 ]]; then
  echo "Expected one paired V12 student and V8 teacher checkpoint" >&2
  exit 3
fi

bash tools/train_light_4090d.sh \
  "${config}" \
  "${work_dir}" \
  --cfg-options \
  "load_from=${student_checkpoints[0]}" \
  "model.teacher_checkpoint=${teacher_checkpoints[0]}" \
  "model.backbone.active_block_indices=[${subset}]" \
  "randomness.seed=${seed}" \
  "randomness.deterministic=True"

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f \
  -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one Phase 25 best checkpoint" >&2
  exit 4
fi

python tools/test.py \
  "${config}" \
  "${checkpoints[0]}" \
  --work-dir "${work_dir}/val_eval" \
  --cfg-options \
  "model.teacher_checkpoint=${teacher_checkpoints[0]}" \
  "model.backbone.active_block_indices=[${subset}]" \
  "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
  "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
  "randomness.seed=${seed}" \
  "randomness.deterministic=True" \
  2>&1 | tee "${work_dir}/val_eval.log"
