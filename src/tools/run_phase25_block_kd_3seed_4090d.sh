#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

phase23_summary="work_dirs/phase23_clean_v12/summary.json"
phase24_summary="work_dirs/phase24_block_screen/summary.json"
output_root="work_dirs/phase25_block_kd"
seeds=(42 123 3407)

mapfile -t candidates < <(python - "${phase24_summary}" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not summary.get("passed") or summary.get("test_evaluated") is not False:
    raise SystemExit("Phase 24 must pass with test sealed before Phase 25")
rows = [row for row in summary["candidates"] if row.get("qualifies")]
for row in sorted(rows, key=lambda item: item["speedup"], reverse=True):
    indices = ",".join(str(value) for value in row["benchmark"]["active_block_indices"])
    print(f"{row['name']}|{indices}")
PY
)

if [[ "${#candidates[@]}" -eq 0 ]]; then
  echo "No Phase 24 candidate qualified; close static block pruning." >&2
  exit 5
fi

for entry in "${candidates[@]}"; do
  candidate="${entry%%|*}"
  subset="${entry#*|}"
  for seed in "${seeds[@]}"; do
    bash tools/train_phase25_block_kd_4090d.sh \
      "${candidate}" "${subset}" "${seed}"
  done

  candidate_root="${output_root}/${candidate}"
  if python tools/summarize_phase25_block_kd.py \
    --phase23-summary "${phase23_summary}" \
    --phase24-summary "${phase24_summary}" \
    --candidate "${candidate}" \
    --root "${candidate_root}" \
    --seeds "${seeds[@]}" \
    --max-run-paired-drop 1.0 \
    --max-mean-paired-drop 0.50 \
    --max-std-miou 0.60 \
    --max-mean-weak-drop 0.50 \
    --output "${candidate_root}/summary.json"; then
    echo "Phase 25 selected ${candidate}. Test remains sealed."
    exit 0
  fi
  echo "Phase 25 rejected ${candidate}; trying the next slower qualified candidate."
done

echo "All qualified static block-pruning candidates failed Phase 25." >&2
exit 1
