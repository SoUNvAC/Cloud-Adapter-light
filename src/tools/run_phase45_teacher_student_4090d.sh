#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase45_teacher_student"
work_dir="${root}/student"
config="configs/protocol/phase45_teacher_student_v8_l8.py"
oracle="work_dirs/phase45_oracle/summary.json"
python - "${oracle}" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("decision") != "proceed_to_factorized_method":
    raise SystemExit("Phase 45 Oracle continuation gate did not pass")
if row.get("target_test_evaluated") is not False:
    raise SystemExit("Target test is not sealed")
PY

if [[ ! -f "${root}/pseudo_summary.json" ]]; then
  python tools/generate_phase45_teacher_pseudo.py
fi
python - "${root}/pseudo_summary.json" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("passed") is not True or row.get("target_labels_read") is not False:
    raise SystemExit("Pseudo-label generation integrity gate failed")
PY

resume_args=()
if [[ -e "${work_dir}" ]]; then
  if [[ -f "${work_dir}/iter_10000.pth" && -f "${work_dir}/TRAIN_COMPLETE" ]]; then
    echo "Reusing completed Phase 45 teacher-student training"
  elif [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint teacher-student directory"
  else
    echo "Refusing checkpoint-bearing teacher-student run without last_checkpoint" >&2
    exit 2
  fi
fi
if [[ ! -f "${work_dir}/TRAIN_COMPLETE" ]]; then
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
    --cfg-options randomness.seed=42 randomness.deterministic=True
  [[ -f "${work_dir}/iter_10000.pth" ]] || exit 3
  touch "${work_dir}/TRAIN_COMPLETE"
fi
python tools/evaluate_phase45_teacher_student.py
