#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

report="${PHASE23_REPAIR_REPORT:-work_dirs/phase23_repair/PHASE23_REPAIR_REPORT.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 23 REPAIR A FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  for path in \
    work_dirs/phase23_repair_lrsweep/summary.json \
    work_dirs/phase23_repair_selected/summary.json; do
    if [[ -f "${path}" ]]; then
      echo "----- ${path} -----"
      python -m json.tool "${path}"
      echo "----- end ${path} -----"
    fi
  done
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
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "===== PHASE 23 REPAIR A ONE-CLICK RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "GIT_STATUS_BEGIN"
git status --short 2>/dev/null || true
echo "GIT_STATUS_END"

python - <<'PY'
import json
from pathlib import Path

path = Path("work_dirs/phase23_clean_v12/summary.json")
row = json.loads(path.read_text(encoding="utf-8"))
expected = row.get("gates", {})
if row.get("passed") is not False:
    raise SystemExit("Repair A requires the completed failed Trial A")
if expected.get("mean_paired_drop") is not False:
    raise SystemExit("Trial A did not fail the preregistered paired-drop gate")
if not all(expected.get(key) is True for key in ("phase22_passed", "all_runs_miou", "mean_miou", "std_miou")):
    raise SystemExit("Trial A failure is not eligible for the one allowed LR repair")
if row.get("test_evaluated") is not False:
    raise SystemExit("Test was not sealed")
print("Trial A failure is eligible for Repair A")
PY

echo
echo "===== SHORT LINEAR SWEEP ====="
bash tools/run_phase23_repair_sweep_4090d.sh

echo
echo "===== FROZEN-MULTIPLIER THREE-SEED CONFIRMATION ====="
bash tools/run_phase23_repair_selected_3seed_4090d.sh

echo
echo "Phase 23 Repair A completed. No later phase was started."
