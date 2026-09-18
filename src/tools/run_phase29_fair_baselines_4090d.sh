#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

phase21="eval_result/phase21_protocol/result.json"
python - "${phase21}" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not result.get("passed"):
    raise SystemExit("Phase 21 clean protocol audit must pass before Phase 29")
PY

root="work_dirs/phase29_fair_baselines"
mkdir -p "${root}"
python tools/audit_phase29_fair_baselines.py \
  --max-method-param-ratio 2.0 \
  --output "${root}/audit.json"

methods=(frozen rein cloud_adapter)
seeds=(42 123 3407)
for seed in "${seeds[@]}"; do
  for method in "${methods[@]}"; do
    bash tools/train_phase29_fair_baseline_4090d.sh "${method}" "${seed}"
  done
done

if python tools/summarize_phase29_fair_baselines.py \
  --audit "${root}/audit.json" \
  --root "${root}" \
  --seeds "${seeds[@]}" \
  --min-run-miou 50.0 \
  --max-std-miou 0.75 \
  --min-cloud-gain-vs-frozen 1.0 \
  --min-cloud-weak-gain-vs-frozen 1.0 \
  --min-cloud-difference-vs-rein -0.50 \
  --output "${root}/summary.json"; then
  echo "Phase 29 passed. Test remains sealed."
else
  echo "Phase 29 failed: do not claim a Cloud-Adapter-specific advantage." >&2
  echo "Switch compression experiments to the strongest fair PEFT baseline." >&2
  exit 1
fi
