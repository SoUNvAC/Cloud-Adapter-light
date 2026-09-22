#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

report="${PHASE45B_ORACLE_REPORT:-work_dirs/phase45_oracle/PHASE45B_ORACLE_CONSOLE.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 45B ORACLE FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f work_dirs/phase45_oracle/summary.json ]]; then
    echo "----- summary.json -----"
    python -m json.tool work_dirs/phase45_oracle/summary.json
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

echo "===== PHASE 45B ORACLE ONE-CLICK RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
git status --short --untracked-files=no
bash tools/run_phase45_oracle_4090d.sh
