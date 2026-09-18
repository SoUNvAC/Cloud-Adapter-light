#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-mixed-fp16}"
case "${mode}" in
  mixed-fp16|int8-qdq) ;;
  *) echo "Mode must be mixed-fp16 or int8-qdq" >&2; exit 2 ;;
esac

artifact_root="${PHASE30_ARTIFACT_ROOT:-work_dirs/phase26_real_int8}"
selection="${artifact_root}/selection.json"
if [[ "${mode}" == "mixed-fp16" ]]; then
  logits_onnx="${artifact_root}/student_fp32.onnx"
else
  logits_onnx="${artifact_root}/student_int8_qdq.onnx"
fi
for path in "${selection}" "${logits_onnx}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing transferred Phase 26 artifact: ${path}" >&2
    exit 3
  fi
done
if [[ "${mode}" == "int8-qdq" ]]; then
  phase26_summary="${artifact_root}/summary.json"
  if [[ ! -f "${phase26_summary}" ]]; then
    echo "INT8 requires the transferred Phase 26 summary: ${phase26_summary}" >&2
    exit 3
  fi
  python - "${phase26_summary}" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("phase") != 26 or summary.get("passed") is not True:
    raise SystemExit("INT8 is closed unless the Phase 26 validation gate passed")
if summary.get("test_evaluated") is not False:
    raise SystemExit("Phase 26 prerequisite must keep CloudSEN12 test sealed")
PY
fi

output_root="work_dirs/phase30_jetson/${mode}"
mkdir -p "${output_root}"

python tools/audit_phase30_jetson.py \
  --required-device-substring "NVIDIA Jetson Orin Nano" \
  --required-power-mode-substring "MAXN" \
  --min-memory-mib 7000 \
  --output "${output_root}/device_audit.json"

mask_onnx="${output_root}/student_mask.onnx"
engine="${output_root}/student_mask.engine"
python tools/prepare_phase20_mask_onnx.py \
  --input "${logits_onnx}" \
  --output "${mask_onnx}" \
  --input-size 512 \
  --force

python tools/build_phase30_jetson_tensorrt.py \
  --onnx "${mask_onnx}" \
  --output "${engine}" \
  --mode "${mode}" \
  --workspace-gib 2.0 \
  --force

if python tools/eval_phase30_jetson.py \
  --audit "${output_root}/device_audit.json" \
  --selection "${selection}" \
  --engine "${engine}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/val \
  --ann-dir data/cloudsen12_high_l1c/ann_dir/val \
  --output "${output_root}/summary.json" \
  --expected-samples 535 \
  --benchmark-samples 20 \
  --warmup 50 \
  --iters 200 \
  --benchmark-repeats 5 \
  --idle-seconds 10 \
  --tegrastats-interval-ms 100 \
  --max-miou-drop 0.50 \
  --max-mean-latency-ms 80.0 \
  --max-p90-latency-ms 100.0 \
  --max-repeat-cv-percent 5.0 \
  --max-energy-j-per-image 2.0 \
  --max-peak-power-w 25.0 \
  --max-temperature-c 80.0 \
  --max-ram-mb 7500; then
  echo "Phase 30 ${mode} target-device gate passed. Test remains sealed."
else
  echo "Phase 30 ${mode} target-device gate failed." >&2
  echo "Do not claim real-time edge deployment; switch to FP16 whole-network channel/token reduction." >&2
  exit 1
fi
