#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
root="work_dirs/phase49_target_factor_adapter"
report="${root}/PHASE49_REPORT.txt"
mkdir -p "${root}"; : > "${report}"; exec > >(tee -a "${report}") 2>&1
finish_report() { status=$?; trap - EXIT; set +e; echo; echo "===== PHASE 49 FINAL STATUS =====";
  echo "PIPELINE_EXIT_CODE=${status}"; [[ -f "${root}/summary.json" ]] && python -m json.tool "${root}/summary.json";
  echo "REPORT_FILE=$(realpath "${report}")"; exit "${status}"; }
trap finish_report EXIT
set -e
python - <<'PY'
import json
from pathlib import Path
row=json.loads(Path("work_dirs/phase48_factor_prototypes/summary.json").read_text())
if row.get("passed") is not False or row.get("decision") != "close_fixed_prototype_inference":
    raise SystemExit("Phase 48 stop-loss decision is required")
if row["gates"].get("target_test_evaluated") is not False:
    raise SystemExit("Target test is not sealed")
PY
echo "===== PHASE 49 TARGET FACTOR ADAPTER ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"; git status --short --untracked-files=no
[[ -f "${root}/final.pth" ]] || python tools/train_phase49_target_factor_adapter.py
python tools/evaluate_phase49_target_factor_adapter.py
