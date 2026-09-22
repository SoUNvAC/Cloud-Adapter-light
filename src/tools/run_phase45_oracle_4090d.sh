#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

audit="work_dirs/phase45_protocol_audit/summary.json"
source_summary="work_dirs/phase45_source_only/summary.json"
python - "${audit}" "${source_summary}" <<'PY'
import json
from pathlib import Path
import sys
audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if audit.get("passed") is not True or source.get("passed") is not True:
    raise SystemExit("Phase 45A and source-only evidence are required")
if source.get("target_test_evaluated") is not False:
    raise SystemExit("Target test is not sealed")
PY

export PHASE29_MOBILENETV2_PRETRAINED="work_dirs/phase29_mobilenetv2_pilot/pretrained/mobilenet_v2_backbone_only.pth"

run_oracle() {
  local name="$1"
  local config="$2"
  local work_dir="work_dirs/phase45_oracle_${name}/seed42"
  local resume_args=()
  if [[ -e "${work_dir}" ]]; then
    mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
    if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/TRAIN_COMPLETE" ]]; then
      echo "Reusing completed Phase 45 oracle: ${name}"
      return
    elif [[ -f "${work_dir}/last_checkpoint" ]]; then
      resume_args=(--resume)
    elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
      echo "Restarting zero-checkpoint oracle directory: ${name}"
    else
      echo "Refusing checkpoint-bearing oracle run without last_checkpoint: ${name}" >&2
      exit 2
    fi
  fi
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
    --cfg-options randomness.seed=42 randomness.deterministic=True
  mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected one best ${name} checkpoint, found ${#checkpoints[@]}" >&2
    exit 3
  fi
  touch "${work_dir}/TRAIN_COMPLETE"
}

run_oracle v8 configs/protocol/phase45_oracle_v8_l8.py
run_oracle compact configs/protocol/phase45_oracle_compact_l8.py
python tools/run_phase45_oracle_eval.py
