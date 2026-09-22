#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

report="${PHASE45B_SOURCE_REPORT:-work_dirs/phase45_source_only/PHASE45B_SOURCE_CONSOLE.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 45B SOURCE-ONLY FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f work_dirs/phase45_source_only/summary.json ]]; then
    echo "----- summary.json -----"
    python -m json.tool work_dirs/phase45_source_only/summary.json
  fi
  report_absolute=$(python - "${report}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
  echo "REPORT_FILE=${report_absolute}"
  exit "${status}"
}
trap finish_report EXIT
set -e

echo "===== PHASE 45B SOURCE-ONLY ONE-CLICK RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
git status --short --untracked-files=no
python tools/run_phase45_source_only.py
