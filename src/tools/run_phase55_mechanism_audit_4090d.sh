#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root="work_dirs/phase55_thin_shadow_mechanism"
mkdir -p "${root}"

set +e
python tools/run_phase55_mechanism_audit.py 2>&1 | tee "${root}/PHASE55_CONSOLE.txt"
status=${PIPESTATUS[0]}
set -e
if [[ -f "${root}/summary.json" ]]; then
  touch "${root}/AUDIT_COMPLETE"
fi
exit "${status}"
