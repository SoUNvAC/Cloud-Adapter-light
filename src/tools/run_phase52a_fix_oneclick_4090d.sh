#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
report="${PHASE52A_FIX_REPORT:-work_dirs/phase52a_fix_shadow_status_1pct/PHASE52A_FIX_CONSOLE.txt}"
mkdir -p "$(dirname "${report}")"
set +e
bash tools/run_phase52a_fix_4090d.sh 2>&1 | tee "${report}"
status=${PIPESTATUS[0]}
set -e
if [[ -f work_dirs/phase52a_fix_shadow_status_1pct/summary.json ]]; then
  python -m json.tool work_dirs/phase52a_fix_shadow_status_1pct/summary.json
fi
exit "${status}"
