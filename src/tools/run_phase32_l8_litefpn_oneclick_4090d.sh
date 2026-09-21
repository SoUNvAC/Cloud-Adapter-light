#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

report="${PHASE32_REPORT:-work_dirs/phase32_l8_litefpn/PHASE32_REPORT.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 32 FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f work_dirs/phase32_l8_litefpn/summary.json ]]; then
    echo "----- summary.json -----"
    python -m json.tool work_dirs/phase32_l8_litefpn/summary.json
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

echo "===== PHASE 32 ONE-CLICK RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
git status --short --untracked-files=no
bash tools/run_phase32_l8_litefpn_4090d.sh
echo "Phase 32 completed. No later phase was started."
