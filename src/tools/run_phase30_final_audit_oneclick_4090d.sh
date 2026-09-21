#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase30_final_audit"
report="${PHASE30_REPORT:-${root}/PHASE30_REPORT.txt}"
mkdir -p "${root}"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 30 FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f "${root}/summary.json" ]]; then
    echo "----- summary.json -----"
    python -m json.tool "${root}/summary.json"
  fi
  if [[ -f "${root}/FINAL_EVIDENCE_REPORT.md" ]]; then
    echo "----- FINAL_EVIDENCE_REPORT.md -----"
    cat "${root}/FINAL_EVIDENCE_REPORT.md"
  fi
  report_absolute=$(python - "${report}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
  echo "REPORT_FILE=${report_absolute}"
  echo "Send this single report file back to Codex."
  exit "${status}"
}
trap finish_report EXIT
set -e

echo "===== PHASE 30 ONE-CLICK FINAL AUDIT ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "TRACKED_GIT_STATUS_BEGIN"
git status --short --untracked-files=no
echo "TRACKED_GIT_STATUS_END"

python -m unittest discover -s tests -p 'test_*.py'
python tools/run_phase30_final_audit.py --root "${root}"

echo
echo "Phase 30 completed. No training or dataset inference was run."
