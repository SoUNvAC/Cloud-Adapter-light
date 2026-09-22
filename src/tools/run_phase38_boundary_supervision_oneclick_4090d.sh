#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

report="${PHASE38_REPORT:-work_dirs/phase38_boundary_supervision/PHASE38_REPORT.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 38 FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f work_dirs/phase38_boundary_supervision/summary.json ]]; then
    echo "----- summary.json -----"
    python -m json.tool work_dirs/phase38_boundary_supervision/summary.json
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
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "===== PHASE 38 ONE-CLICK RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
git status --short --untracked-files=no
bash tools/run_phase38_boundary_supervision_4090d.sh
echo "Phase 38 completed. No later phase was started."
