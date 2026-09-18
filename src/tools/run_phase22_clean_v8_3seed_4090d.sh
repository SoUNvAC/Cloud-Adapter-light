#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seeds=(42 123 3407)
for seed in "${seeds[@]}"; do
  bash tools/train_phase22_clean_v8_4090d.sh "${seed}"
done

python tools/summarize_phase22_clean_v8.py \
  --root work_dirs/phase22_clean_v8 \
  --seeds "${seeds[@]}" \
  --min-run-miou 69.0 \
  --min-mean-miou 69.5 \
  --max-std-miou 0.50 \
  --output work_dirs/phase22_clean_v8/summary.json
