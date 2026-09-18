#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source tools/activate_onnxruntime_cuda_libs.sh

phase25_root="work_dirs/phase25_block_kd"
output_root="work_dirs/phase26_real_int8"
selection="${output_root}/selection.json"
config="configs/protocol/phase23_clean_v12_l1c.py"
reference_onnx="${output_root}/student_fp32.onnx"
qdq_onnx="${output_root}/student_int8_qdq.onnx"
fp16_engine="${output_root}/student_fp16.engine"
int8_engine="${output_root}/student_int8_qdq.engine"

mkdir -p "${output_root}"
python tools/select_phase26_checkpoint.py \
  --phase25-root "${phase25_root}" \
  --output "${selection}"

mapfile -t selected < <(python - "${selection}" <<'PY'
import json
from pathlib import Path
import sys

row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(row["checkpoint"])
print(row["reference_validation_mIoU"])
print(" ".join(str(value) for value in row["active_block_indices"]))
PY
)
checkpoint="${selected[0]}"
reference_miou="${selected[1]}"
read -r -a active_blocks <<< "${selected[2]}"

python tools/export_phase12_onnx.py \
  --config "${config}" \
  --checkpoint "${checkpoint}" \
  --output "${reference_onnx}" \
  --precision fp32 \
  --active-block-indices "${active_blocks[@]}" \
  --force

python tools/quantize_phase26_int8_qdq.py \
  --onnx "${reference_onnx}" \
  --output "${qdq_onnx}" \
  --calibration-image-dir data/cloudsen12_high_l1c/img_dir/val \
  --calibration-samples 256 \
  --force

python tools/build_phase12_tensorrt.py \
  --onnx "${reference_onnx}" \
  --output "${fp16_engine}" \
  --mode mixed-fp16 \
  --force

python tools/build_phase12_tensorrt.py \
  --onnx "${qdq_onnx}" \
  --output "${int8_engine}" \
  --mode int8-qdq \
  --force

if python tools/eval_phase26_int8_val.py \
  --reference-onnx "${reference_onnx}" \
  --qdq-onnx "${qdq_onnx}" \
  --fp16-engine "${fp16_engine}" \
  --int8-engine "${int8_engine}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/val \
  --ann-dir data/cloudsen12_high_l1c/ann_dir/val \
  --expected-reference-miou "${reference_miou}" \
  --max-reference-miou-delta 0.05 \
  --max-quantized-miou-drop 0.50 \
  --min-quantized-agreement 98.50 \
  --min-int8-speedup 1.15 \
  --output "${output_root}/summary.json"; then
  echo "Phase 26 passed. Test remains sealed; proceed to target-device replication."
else
  echo "Phase 26 failed its preregistered val gate." >&2
  echo "Close PTQ INT8 and switch to FP16 structured channel/token reduction; do not report simulated INT8." >&2
  exit 1
fi
