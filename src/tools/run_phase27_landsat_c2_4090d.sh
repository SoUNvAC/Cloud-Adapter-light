#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source tools/activate_onnxruntime_cuda_libs.sh

data_root="${LANDSAT8_C2_CCA_ROOT:-data/landsat8_c2_cca}"
manifest="${data_root}/manifest.json"
output_root="work_dirs/phase27_landsat8_c2"
selection="${output_root}/selection.json"
audit="${output_root}/audit.json"
onnx="${output_root}/phase25_student_fp32.onnx"
config="configs/protocol/phase23_clean_v12_l1c.py"

mkdir -p "${output_root}"

python - <<'PY'
try:
    import rasterio
except ImportError as error:
    raise SystemExit(
        "Phase 27 requires rasterio. Run: "
        "python -m pip install -r requirements-external.txt"
    ) from error
print(f"Rasterio: {rasterio.__version__}")
PY

python tools/audit_phase27_landsat_c2.py \
  --root "${data_root}" \
  --manifest "${manifest}" \
  --expected-scenes 48 \
  --output "${audit}"

python tools/select_phase26_checkpoint.py \
  --phase25-root work_dirs/phase25_block_kd \
  --output "${selection}"

mapfile -t selected < <(python - "${selection}" <<'PY'
import json
from pathlib import Path
import sys

row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(row["checkpoint"])
print(" ".join(str(value) for value in row["active_block_indices"]))
PY
)
checkpoint="${selected[0]}"
read -r -a active_blocks <<< "${selected[1]}"

python tools/export_phase12_onnx.py \
  --config "${config}" \
  --checkpoint "${checkpoint}" \
  --output "${onnx}" \
  --precision fp32 \
  --active-block-indices "${active_blocks[@]}" \
  --force

if python tools/eval_phase27_landsat_c2.py \
  --audit "${audit}" \
  --data-root "${data_root}" \
  --onnx "${onnx}" \
  --output "${output_root}/summary.json" \
  --expected-scenes 48 \
  --tile-size 512 \
  --core-size 384 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260918 \
  --min-cloud-iou 50.0 \
  --min-cloud-f1 65.0 \
  --min-thin-recall 40.0 \
  --min-scene-macro-iou-ci-low 40.0 \
  --min-mean-difference-vs-cfmask -10.0; then
  echo "Phase 27 passed. CloudSEN12 test remains sealed."
else
  echo "Phase 27 failed its frozen zero-shot cross-sensor gate." >&2
  echo "Switch to Phase 28: train a sensor-input adapter on 38-Cloud train scenes only, then retest unchanged USGS C2 scenes." >&2
  exit 1
fi
