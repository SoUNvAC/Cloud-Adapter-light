#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="work_dirs/phase28_l8_external"
phase27_summary="work_dirs/phase27_resnet18_3seed/summary.json"
v8_config="configs/protocol/phase28_l8_v8_l1c.py"
compact_config="configs/protocol/phase28_l8_resnet18_l1c.py"
export PHASE26_RESNET18_PRETRAINED="work_dirs/phase26_resnet18_pilot/pretrained/resnet18_v1c-b5776b93.pth"
export PHASE28_L8_ROOT="data/l8_biome"

python - "${phase27_summary}" <<'PY'
import json
from pathlib import Path
import sys
row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if row.get("phase") != 27 or row.get("passed") is not True or row.get("test_evaluated") is not False:
    raise SystemExit("Phase 28 requires passed, sealed-internal-test Phase 27")
PY
if [[ ! -d "${PHASE28_L8_ROOT}/img_dir/test" || ! -d "${PHASE28_L8_ROOT}/ann_dir/test" ]]; then
  echo "L8 Biome external test split is unavailable" >&2
  exit 6
fi

mkdir -p "${root}/v8" "${root}/resnet18"
for seed in 42 123 3407; do
  v8_checkpoint="work_dirs/phase22_clean_v8/seed${seed}/best_mIoU_iter_40000.pth"
  if [[ ! -f "${root}/v8/seed${seed}.complete" ]]; then
    python tools/test.py "${v8_config}" "${v8_checkpoint}" \
      --work-dir "${root}/v8/eval_seed${seed}" \
      --cfg-options "randomness.seed=${seed}" "randomness.deterministic=True" \
      2>&1 | tee "${root}/v8/seed${seed}.log"
    touch "${root}/v8/seed${seed}.complete"
  fi

  if [[ "${seed}" == "42" ]]; then
    compact_dir="work_dirs/phase26_resnet18_pilot/seed42"
  else
    compact_dir="work_dirs/phase27_resnet18_3seed/seed${seed}"
  fi
  mapfile -t compact_checkpoints < <(find "${compact_dir}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -print)
  if [[ "${#compact_checkpoints[@]}" -ne 1 ]]; then
    echo "Expected one compact checkpoint for seed ${seed}" >&2
    exit 4
  fi
  if [[ ! -f "${root}/resnet18/seed${seed}.complete" ]]; then
    python tools/test.py "${compact_config}" "${compact_checkpoints[0]}" \
      --work-dir "${root}/resnet18/eval_seed${seed}" \
      --cfg-options "randomness.seed=${seed}" "randomness.deterministic=True" \
      2>&1 | tee "${root}/resnet18/seed${seed}.log"
    touch "${root}/resnet18/seed${seed}.complete"
  fi
done

python tools/summarize_phase28_l8_external.py \
  --root "${root}" --min-compact-miou 35.0 \
  --max-mean-drop 6.5 --max-compact-std 1.0 \
  --output "${root}/summary.json"
