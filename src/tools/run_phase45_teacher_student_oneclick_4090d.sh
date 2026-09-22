#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
report="${PHASE45_TS_REPORT:-work_dirs/phase45_teacher_student/PHASE45_TS_CONSOLE.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1
finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 45 TEACHER-STUDENT FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  test -f work_dirs/phase45_teacher_student/summary.json && python -m json.tool work_dirs/phase45_teacher_student/summary.json
  exit "${status}"
}
trap finish_report EXIT
set -e
echo "===== PHASE 45 TEACHER-STUDENT ONE-CLICK RUN ====="
echo "GIT_COMMIT=$(git rev-parse HEAD)"
git status --short --untracked-files=no
bash tools/run_phase45_teacher_student_4090d.sh
