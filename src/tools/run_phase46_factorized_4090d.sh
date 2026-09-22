#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase46_factorized_v8"
work_dir="${root}/seed42"
config="configs/protocol/phase46_factorized_v8_l1c.py"
report="${root}/PHASE46_REPORT.txt"
mkdir -p "${root}"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 46 FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f "${root}/summary.json" ]]; then
    python -m json.tool "${root}/summary.json"
  fi
  echo "REPORT_FILE=$(realpath "${report}")"
  exit "${status}"
}
trap finish_report EXIT
set -e

python - <<'PY'
import json
from pathlib import Path
oracle = json.loads(Path("work_dirs/phase45_oracle/summary.json").read_text())
pseudo = json.loads(Path("work_dirs/phase45_teacher_student/pseudo_summary.json").read_text())
if oracle.get("decision") != "proceed_to_factorized_method":
    raise SystemExit("Phase 45 oracle continuation gate did not pass")
if pseudo.get("passed") is not False:
    raise SystemExit("Expected the preregistered fixed-teacher baseline to fail")
if oracle.get("target_test_evaluated") is not False:
    raise SystemExit("Target test is not sealed")
PY

echo "===== PHASE 46 FACTORIZED OUTPUT RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
git status --short --untracked-files=no

resume_args=()
if [[ -e "${work_dir}" ]]; then
  mapfile -t completed < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#completed[@]}" -eq 1 && -f "${work_dir}/TRAIN_COMPLETE" ]]; then
    echo "Reusing completed Phase 46 training"
  elif [[ -f "${work_dir}/last_checkpoint" ]]; then
    resume_args=(--resume)
  elif ! find "${work_dir}" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
    echo "Restarting zero-checkpoint Phase 46 directory"
  else
    echo "Refusing checkpoint-bearing Phase 46 run without last_checkpoint" >&2
    exit 2
  fi
fi
if [[ ! -f "${work_dir}/TRAIN_COMPLETE" ]]; then
  bash tools/train_light_4090d.sh "${config}" "${work_dir}" "${resume_args[@]}" \
    --cfg-options randomness.seed=42 randomness.deterministic=True
  mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  [[ "${#checkpoints[@]}" -eq 1 ]] || { echo "Expected one best checkpoint" >&2; exit 3; }
  touch "${work_dir}/TRAIN_COMPLETE"
fi
mapfile -t checkpoints < <(find "${work_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
[[ "${#checkpoints[@]}" -eq 1 ]] || exit 4
python tools/evaluate_phase46_factorized.py --checkpoint "${checkpoints[0]}"
