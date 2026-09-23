#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase52b_shadow_residual_1pct"
config="configs/protocol/phase52b_shadow_residual_1pct_l1c.py"
work_dir="${root}/seed52"
mkdir -p "${root}"

python - <<'PY'
import json
from pathlib import Path
phase52 = json.loads(Path("work_dirs/phase52_sparse_msre_1pct/summary.json").read_text())
diagnosis = json.loads(Path("work_dirs/phase52_shadow_diagnosis/summary.json").read_text())
if phase52.get("decision") != "stop_phase52_msre_route":
    raise SystemExit("Frozen Phase 52A result is required")
if diagnosis["phase52_shadow_flow"]["predicted_shadow_fraction"] >= 0.01:
    raise SystemExit("Diagnosis does not support shadow collapse")
if diagnosis["target_test_evaluated"] or diagnosis["cloudsen_internal_test_evaluated"]:
    raise SystemExit("A sealed test set was evaluated")
PY

python tools/audit_phase52b_shadow_residual.py

if [[ ! -f "${work_dir}/TRAIN_COMPLETE" ]]; then
  resume_args=()
  if [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  fi
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}"
  touch "${work_dir}/TRAIN_COMPLETE"
fi

mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected one best Phase 52B checkpoint, found ${#checkpoints[@]}" >&2
  exit 3
fi
python tools/evaluate_phase52b_shadow_residual.py --checkpoint "${checkpoints[0]}"
