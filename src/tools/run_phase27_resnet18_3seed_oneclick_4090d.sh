#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

report="${PHASE27_REPORT:-work_dirs/phase27_resnet18_3seed/PHASE27_REPORT.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 27 FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f work_dirs/phase27_resnet18_3seed/summary.json ]]; then
    echo "----- summary.json -----"
    python -m json.tool work_dirs/phase27_resnet18_3seed/summary.json
    echo "----- end summary.json -----"
  else
    echo "summary.json was not produced; inspect the error above."
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

echo "===== PHASE 27 ONE-CLICK RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "GIT_STATUS_BEGIN"
git status --short 2>/dev/null || true
echo "GIT_STATUS_END"

bash tools/run_phase27_resnet18_3seed_4090d.sh

echo
echo "Phase 27 completed. No later phase was started."
