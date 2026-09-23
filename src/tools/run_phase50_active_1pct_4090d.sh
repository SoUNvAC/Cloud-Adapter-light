#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase50_active_1pct"
config="configs/protocol/phase50_active_1pct_v8_l1c.py"
work_dir="${root}/seed50"
mkdir -p "${root}"

python - <<'PY'
import json
from pathlib import Path
row = json.loads(Path("work_dirs/phase49_target_factor_adapter/summary.json").read_text())
if row.get("decision") != "stop_adjacent_da_variants":
    raise SystemExit("Phase 49 terminal DA stop is required")
if row.get("target_test_evaluated") is not False:
    raise SystemExit("Target test seal was violated")
PY

if [[ ! -f "${root}/selection.json" ]]; then
  python tools/select_phase50_active_1pct.py
else
  echo "Reusing frozen Phase 50 selection"
fi

resume_args=()
if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/TRAIN_COMPLETE" ]]; then
    echo "Reusing completed Phase 50 training"
  elif [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint Phase 50 directory"
  else
    echo "Refusing checkpoint-bearing Phase 50 run without last_checkpoint" >&2
    exit 2
  fi
fi

if [[ ! -f "${work_dir}/TRAIN_COMPLETE" ]]; then
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}"
  touch "${work_dir}/TRAIN_COMPLETE"
fi

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected one best Phase 50 checkpoint, found ${#checkpoints[@]}" >&2
  exit 3
fi
python tools/evaluate_phase50_active_1pct.py --checkpoint "${checkpoints[0]}"
