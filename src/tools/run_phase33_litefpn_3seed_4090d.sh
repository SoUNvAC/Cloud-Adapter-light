#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase33_litefpn_3seed"
phase31_root="work_dirs/phase31_mobilenetv2_litefpn"
phase31_summary="${phase31_root}/summary.json"
phase32_summary="work_dirs/phase32_l8_litefpn/summary.json"

python - "${phase31_summary}" "${phase32_summary}" <<'PY'
import json
from pathlib import Path
import sys
p31 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
p32 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if p31.get("phase") != 31 or p31.get("passed") is not True:
    raise SystemExit("Phase 33 requires passed Phase 31")
if p32.get("phase") != 32 or p32.get("passed") is not True:
    raise SystemExit("Phase 33 requires passed Phase 32")
if p31.get("internal_test_evaluated") is not False or p32.get("internal_test_evaluated") is not False:
    raise SystemExit("CloudSEN internal test is not sealed")
PY

for seed in 123 3407; do
  bash tools/train_phase33_litefpn_seed_4090d.sh "${seed}"
done

python tools/summarize_phase33_litefpn_3seed.py \
  --phase31-summary "${phase31_summary}" --phase31-root "${phase31_root}" \
  --root "${root}" --min-run-miou 64.5 --min-mean-miou 65.3 \
  --max-std-miou 0.75 --output "${root}/summary.json"
