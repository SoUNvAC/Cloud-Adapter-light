#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase53_shadow_aux_msre_1pct"
config="configs/protocol/phase53_shadow_aux_msre_1pct_l1c.py"
work_dir="${root}/seed52"
mkdir -p "${root}"

python - <<'PY'
import json
from pathlib import Path
row = json.loads(Path("work_dirs/phase52_sparse_msre_1pct/summary.json").read_text())
if row.get("decision") != "stop_phase52_msre_route":
    raise SystemExit("Frozen Phase 52A stop decision is required")
if row.get("target_test_evaluated") is not False or row.get("cloudsen_internal_test_evaluated") is not False:
    raise SystemExit("A sealed test set was evaluated")
PY

python tools/audit_phase52_sparse_msre.py \
  --config "${config}" --output "${root}/preflight.json"

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
  echo "Expected one best Phase 53 checkpoint, found ${#checkpoints[@]}" >&2
  exit 3
fi
python tools/evaluate_phase53_shadow_aux.py --checkpoint "${checkpoints[0]}"
