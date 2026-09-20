#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mult="${1:?usage: train_phase23_repair_sweep_4090d.sh MULTIPLIER}"
case "${mult}" in
  1|2|3|4) ;;
  *) echo "Multiplier must be one of 1, 2, 3, 4" >&2; exit 2 ;;
esac

config="configs/protocol/phase23_repair_lrsweep_l1c.py"
source_checkpoint="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth"
work_dir="work_dirs/phase23_repair_lrsweep/mult${mult}_seed42"
resume_args=()
export PHASE23_FPN_LR_MULT="${mult}"

if [[ ! -f "${source_checkpoint}" ]]; then
  echo "Missing Phase 22 seed-42 source checkpoint" >&2
  exit 3
fi

if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/val_eval.log" && -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
    echo "Reusing completed Phase 23 repair sweep: multiplier ${mult}"
    exit 0
  fi
  if [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Refusing checkpoint-bearing sweep without last_checkpoint: ${work_dir}" >&2
    exit 4
  fi
fi

bash tools/train_light_4090d.sh \
  "${config}" "${work_dir}" "${resume_args[@]}" \
  --cfg-options "load_from=${source_checkpoint}" "randomness.seed=42" "randomness.deterministic=True"

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one sweep checkpoint, found ${#checkpoints[@]}" >&2
  exit 5
fi

python tools/test.py "${config}" "${checkpoints[0]}" \
  --work-dir "${work_dir}/val_eval" \
  --cfg-options \
  "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
  "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
  "randomness.seed=42" "randomness.deterministic=True" \
  2>&1 | tee "${work_dir}/val_eval.log"
touch "${work_dir}/VAL_EVAL_COMPLETE"
