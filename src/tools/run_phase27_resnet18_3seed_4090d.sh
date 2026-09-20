#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

phase26_summary="work_dirs/phase26_resnet18_pilot/summary.json"
python - "${phase26_summary}" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("phase") != 26 or row.get("passed") is not True or row.get("test_evaluated") is not False:
    raise SystemExit("Phase 27 requires passed, sealed-test Phase 26")
PY

for seed in 123 3407; do
  bash tools/train_phase27_resnet18_seed_4090d.sh "${seed}"
done

python tools/summarize_phase27_resnet18_3seed.py \
  --phase26-summary "${phase26_summary}" \
  --phase26-root work_dirs/phase26_resnet18_pilot \
  --root work_dirs/phase27_resnet18_3seed \
  --seeds 42 123 3407 \
  --min-run-miou 67.0 --min-mean-miou 67.8 --max-std-miou 0.60 \
  --output work_dirs/phase27_resnet18_3seed/summary.json
