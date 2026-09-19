#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

report="${PHASE22_REPORT:-work_dirs/phase22_clean_v8/PHASE22_REPORT.txt}"
mkdir -p "$(dirname "${report}")"
: > "${report}"
exec > >(tee -a "${report}") 2>&1

finish_report() {
  status=$?
  trap - EXIT
  set +e
  echo
  echo "===== PHASE 22 FINAL STATUS ====="
  echo "PIPELINE_EXIT_CODE=${status}"
  if [[ -f work_dirs/phase22_clean_v8/summary.json ]]; then
    echo "----- summary.json -----"
    python -m json.tool work_dirs/phase22_clean_v8/summary.json
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

echo "===== PHASE 22 ONE-CLICK RUN ====="
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
echo "===== PREREQUISITE: PHASE 21 ====="
python - <<'PY'
import json
from pathlib import Path

path = Path("eval_result/phase21_protocol/result.json")
if not path.is_file():
    raise SystemExit("Missing committed Phase 21 audit result")
row = json.loads(path.read_text(encoding="utf-8"))
if row.get("phase") != 21 or row.get("passed") is not True:
    raise SystemExit("Phase 21 audit has not passed")
print("Phase 21 audit: PASSED")
print("Counts:", row.get("counts"))
PY

if [[ -d data/cloudsen12_high_l1c ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="data/cloudsen12_high_l1c"
elif [[ -d ../data/cloudsen12_high_l1c ]]; then
  export CLOUD_ADAPTER_DATA_ROOT="../data/cloudsen12_high_l1c"
else
  echo "Dataset missing; preparing and auditing CloudSEN12 High L1C now."
  bash tools/run_phase21_protocol_audit_4090d.sh
  export CLOUD_ADAPTER_DATA_ROOT="data/cloudsen12_high_l1c"
fi
echo "CLOUD_ADAPTER_DATA_ROOT=${CLOUD_ADAPTER_DATA_ROOT}"

if [[ ! -f checkpoints/dinov2_s_converted_512x512.pth ]]; then
  echo
  echo "===== PREPARE DINOv2-S CHECKPOINT ====="
  bash tools/prepare_dinov2_small.sh
fi

echo
echo "===== PHASE 22 TRAIN / VALIDATE / SUMMARIZE ====="
bash tools/run_phase22_clean_v8_3seed_4090d.sh

echo
echo "Phase 22 completed. No later phase was started."
