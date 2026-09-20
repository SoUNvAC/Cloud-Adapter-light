#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

for mult in 1 2 3 4; do
  bash tools/train_phase23_repair_sweep_4090d.sh "${mult}"
done

python tools/summarize_phase23_repair_sweep.py \
  --root work_dirs/phase23_repair_lrsweep \
  --multipliers 1 2 3 4 \
  --baseline-8k-miou 67.58 \
  --min-improvement 0.25 \
  --output work_dirs/phase23_repair_lrsweep/summary.json
