#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="work_dirs/phase51_trust_audit"
report="${PHASE51_REPORT:-${root}/PHASE51_CONSOLE.txt}"
mkdir -p "${root}"

set +e
python tools/run_phase51_trust_audit.py 2>&1 | tee "${report}"
status=${PIPESTATUS[0]}
set -e
if [[ -f "${root}/summary.json" ]]; then
  python -m json.tool "${root}/summary.json"
fi
exit "${status}"
