#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
report="${PHASE50_REPORT:-work_dirs/phase50_active_1pct/PHASE50_CONSOLE.txt}"
mkdir -p "$(dirname "${report}")"

set +e
bash tools/run_phase50_active_1pct_4090d.sh 2>&1 | tee "${report}"
status=${PIPESTATUS[0]}
set -e

if [[ -f work_dirs/phase50_active_1pct/summary.json ]]; then
  python -m json.tool work_dirs/phase50_active_1pct/summary.json
fi
exit "${status}"
