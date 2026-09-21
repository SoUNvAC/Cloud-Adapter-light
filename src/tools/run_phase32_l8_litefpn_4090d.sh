#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase32_l8_litefpn"
phase31_root="work_dirs/phase31_mobilenetv2_litefpn"
phase31_summary="${phase31_root}/summary.json"
phase28_summary="work_dirs/phase28_l8_external/summary.json"
config="configs/protocol/phase32_l8_mobilenetv2_litefpn_l1c.py"
log="${root}/eval.log"
export PHASE29_MOBILENETV2_PRETRAINED="work_dirs/phase29_mobilenetv2_pilot/pretrained/mobilenet_v2_backbone_only.pth"
export PHASE28_L8_ROOT="data/l8_biome"

python - "${phase31_summary}" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("phase") != 31 or row.get("passed") is not True:
    raise SystemExit("Phase 32 requires passed Phase 31")
if row.get("internal_test_evaluated") is not False:
    raise SystemExit("CloudSEN internal test is not sealed")
PY
if [[ ! -d "${PHASE28_L8_ROOT}/img_dir/test" || ! -d "${PHASE28_L8_ROOT}/ann_dir/test" ]]; then
  echo "L8 Biome external test split is unavailable" >&2
  exit 6
fi
mapfile -t checkpoints < <(find "${phase31_root}/seed42" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one Phase 31 best checkpoint" >&2
  exit 4
fi

mkdir -p "${root}"
if [[ ! -f "${root}/EVAL_COMPLETE" ]]; then
  python tools/test.py "${config}" "${checkpoints[0]}" \
    --work-dir "${root}/eval" \
    --cfg-options "randomness.seed=42" "randomness.deterministic=True" \
    2>&1 | tee "${log}"
  touch "${root}/EVAL_COMPLETE"
fi

python tools/summarize_phase32_l8_litefpn.py \
  --log "${log}" --phase31-summary "${phase31_summary}" \
  --phase28-summary "${phase28_summary}" --min-miou 35.0 \
  --max-drop 6.5 --expected-images 2643 --output "${root}/summary.json"
