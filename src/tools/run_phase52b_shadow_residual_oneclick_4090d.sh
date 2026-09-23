#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
report="${PHASE52B_REPORT:-work_dirs/phase52b_shadow_residual_1pct/PHASE52B_CONSOLE.txt}"
mkdir -p "$(dirname "${report}")"
set +e
bash tools/run_phase52b_shadow_residual_4090d.sh 2>&1 | tee "${report}"
status=${PIPESTATUS[0]}
set -e
if [[ -f work_dirs/phase52b_shadow_residual_1pct/summary.json ]]; then
  python -m json.tool work_dirs/phase52b_shadow_residual_1pct/summary.json
fi
exit "${status}"
