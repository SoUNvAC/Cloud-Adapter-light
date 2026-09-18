#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seeds=(42 123 3407)
phase22_summary="work_dirs/phase22_clean_v8/summary.json"

for seed in "${seeds[@]}"; do
  bash tools/train_phase23_clean_v12_4090d.sh "${seed}"
done

python tools/summarize_phase23_clean_v12.py \
  --phase22-summary "${phase22_summary}" \
  --root work_dirs/phase23_clean_v12 \
  --seeds "${seeds[@]}" \
  --min-run-miou 66.5 \
  --min-mean-miou 67.0 \
  --max-std-miou 0.60 \
  --max-mean-paired-drop 3.0 \
  --output work_dirs/phase23_clean_v12/summary.json
