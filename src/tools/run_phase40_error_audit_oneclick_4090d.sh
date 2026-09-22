#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase40_error_audit"
if [[ -f "${root}/summary.json" ]]; then
  cat "${root}/summary.json"
  exit 0
fi

python - <<'PY'
import json
from pathlib import Path
p39 = json.loads(Path("work_dirs/phase39_pareto_audit/summary.json").read_text())
if p39.get("passed") is not True or p39.get("qualified_candidate") is not False:
    raise SystemExit("Phase 40 requires the completed no-candidate Phase 39 audit")
if p39.get("internal_test_evaluated") is not False:
    raise SystemExit("CloudSEN internal test is not sealed")
PY

checkpoint=(work_dirs/phase37_weakclass_supervision/seed42/best_mIoU_iter_*.pth)
if [[ "${#checkpoint[@]}" -ne 1 || ! -f "${checkpoint[0]}" ]]; then exit 4; fi
if [[ -d "data/cloudsen12_high_l1c" ]]; then
  data_root="data/cloudsen12_high_l1c"
elif [[ -d "../data/cloudsen12_high_l1c" ]]; then
  data_root="../data/cloudsen12_high_l1c"
else
  exit 6
fi

python tools/run_phase40_error_audit.py \
  --config configs/protocol/phase37_mobilenetv2_litefpn_weakclass_l1c.py \
  --checkpoint "${checkpoint[0]}" \
  --image-dir "${data_root}/img_dir/val" \
  --ann-dir "${data_root}/ann_dir/val" \
  --output-dir "${root}" \
  --expected-images 535 --expected-miou 67.35 --max-miou-delta 0.05 \
  --boundary-gap 8.0 --prevalence-gap 8.0
