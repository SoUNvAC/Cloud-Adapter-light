#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

report="${PHASE23_REPORT:-work_dirs/phase23_clean_v12/PHASE23_REPORT.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 23 FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f work_dirs/phase23_clean_v12/summary.json ]]; then
    echo "----- summary.json -----"
    python -m json.tool work_dirs/phase23_clean_v12/summary.json
    echo "----- end summary.json -----"
  else
    echo "summary.json was not produced; inspect the error above."
  fi
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

echo "===== PHASE 23 ONE-CLICK RUN ====="
echo "STARTED_AT=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "WORKING_DIRECTORY=$(pwd)"
echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "GIT_STATUS_BEGIN"
git status --short 2>/dev/null || true
echo "GIT_STATUS_END"
echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
python --version || true
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader 2>/dev/null || true

echo
echo "===== PREREQUISITE: PHASE 22 ====="
python - <<'PY'
import json
from pathlib import Path

path = Path("work_dirs/phase22_clean_v8/summary.json")
if not path.is_file():
    raise SystemExit("Missing Phase 22 summary")
row = json.loads(path.read_text(encoding="utf-8"))
if row.get("phase") != 22 or row.get("passed") is not True:
    raise SystemExit("Phase 22 gate has not passed")
if row.get("selection_split") != "val" or row.get("test_evaluated") is not False:
    raise SystemExit("Phase 22 did not keep the test split sealed")
print("Phase 22 clean baseline: PASSED")
print("Mean mIoU:", row.get("mean_mIoU"))
print("Sample std:", row.get("std_mIoU"))
PY

if [[ ! -f checkpoints/dinov2_s_converted_512x512.pth ]]; then
  echo "Missing required DINOv2-S checkpoint." >&2
  exit 5
fi

echo
echo "===== PHASE 23 TRAIN / VALIDATE / SUMMARIZE ====="
bash tools/run_phase23_clean_v12_3seed_4090d.sh

echo
echo "Phase 23 completed. No later phase was started."
