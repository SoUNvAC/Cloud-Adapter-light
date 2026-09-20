#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seed="${1:-42}"
selection="work_dirs/phase23_repair_lrsweep/summary.json"
config="configs/protocol/phase23_repair_selected_l1c.py"
source_checkpoint="work_dirs/phase22_clean_v8/seed${seed}/best_mIoU_iter_40000.pth"
work_dir="work_dirs/phase23_repair_selected/seed${seed}"
resume_args=()

selected=$(python - "${selection}" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("passed") is not True or row.get("test_evaluated") is not False:
    raise SystemExit("Phase 23 repair sweep has no qualified sealed-test candidate")
print(row["selected_multiplier"])
PY
)
export PHASE23_FPN_LR_MULT="${selected}"

if [[ ! -f "${source_checkpoint}" ]]; then
  echo "Missing paired Phase 22 checkpoint for seed ${seed}" >&2
  exit 3
fi

if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/val_eval.log" && -f "${work_dir}/VAL_EVAL_COMPLETE" ]]; then
    echo "Reusing completed Phase 23 selected repair: seed ${seed}"
    exit 0
  fi
  if [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Refusing checkpoint-bearing repair without last_checkpoint: ${work_dir}" >&2
    exit 4
  fi
fi

echo "PHASE23_SELECTED_FPN_LR_MULT=${PHASE23_FPN_LR_MULT}"
bash tools/train_light_4090d.sh \
  "${config}" "${work_dir}" "${resume_args[@]}" \
  --cfg-options "load_from=${source_checkpoint}" "randomness.seed=${seed}" "randomness.deterministic=True"

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one selected-repair checkpoint, found ${#checkpoints[@]}" >&2
  exit 5
fi

python tools/test.py "${config}" "${checkpoints[0]}" \
  --work-dir "${work_dir}/val_eval" \
  --cfg-options \
  "test_dataloader.dataset.data_prefix.img_path=img_dir/val" \
  "test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/val" \
  "randomness.seed=${seed}" "randomness.deterministic=True" \
  2>&1 | tee "${work_dir}/val_eval.log"
touch "${work_dir}/VAL_EVAL_COMPLETE"
