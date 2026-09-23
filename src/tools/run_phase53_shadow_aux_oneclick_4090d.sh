#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
report="${PHASE53_REPORT:-work_dirs/phase53_shadow_aux_msre_1pct/PHASE53_CONSOLE.txt}"
mkdir -p "$(dirname "${report}")"
set +e
bash tools/run_phase53_shadow_aux_4090d.sh 2>&1 | tee "${report}"
status=${PIPESTATUS[0]}
set -e
if [[ -f work_dirs/phase53_shadow_aux_msre_1pct/summary.json ]]; then
  python -m json.tool work_dirs/phase53_shadow_aux_msre_1pct/summary.json
fi
exit "${status}"
