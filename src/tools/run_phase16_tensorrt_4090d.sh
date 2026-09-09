#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

source tools/activate_onnxruntime_cuda_libs.sh

config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"
checkpoint_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
ort_onnx_path="work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx"
output_dir="work_dirs/phase16_v12_tensorrt"
trt_onnx_path="${output_dir}/v12_export_fpn_fp32_512.onnx"
trt_mode="${TRT_MODE:-mixed-fp16}"
engine_path="${output_dir}/v12_export_fpn_${trt_mode}_b1_512.engine"
checkpoints=("${checkpoint_dir}"/best_mIoU_iter_*.pth)

if ((${#checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${checkpoint_dir}, found ${#checkpoints[@]}"
  exit 1
fi
if [[ ! -f "${ort_onnx_path}" ]]; then
  echo "Missing ${ort_onnx_path}; run tools/export_phase14_onnx_4090d.sh first"
  exit 1
fi
if [[ "${trt_mode}" != "mixed-fp16" && "${trt_mode}" != "fp32" ]]; then
  echo "TRT_MODE must be mixed-fp16 or fp32, got: ${trt_mode}"
  exit 2
fi

python - <<'PY'
try:
    import tensorrt as trt
except ImportError as error:
    raise SystemExit(
        "Missing TensorRT. Install it with: "
        "bash tools/install_tensorrt_cu12.sh"
    ) from error
if not trt.__version__.startswith("10."):
    raise SystemExit(f"Expected TensorRT 10.x, found {trt.__version__}")
print("TensorRT:", trt.__version__)
PY

mkdir -p "${output_dir}"
if [[ ! -f "${trt_onnx_path}" ]]; then
  echo "Exporting FP32 source graph for numerically controlled TensorRT build..."
  python tools/export_phase12_onnx.py \
    --config "${config}" \
    --checkpoint "${checkpoints[0]}" \
    --output "${trt_onnx_path}" \
    --precision fp32 \
    --input-size 512 \
    --opset 17
fi

if [[ "${REBUILD_ENGINE:-0}" == "1" ]]; then
  python tools/build_phase12_tensorrt.py \
    --onnx "${trt_onnx_path}" \
    --output "${engine_path}" \
    --mode "${trt_mode}" \
    --workspace-gib 4 \
    --force
elif [[ ! -f "${engine_path}" ]]; then
  python tools/build_phase12_tensorrt.py \
    --onnx "${trt_onnx_path}" \
    --output "${engine_path}" \
    --mode "${trt_mode}" \
    --workspace-gib 4
else
  echo "Reusing existing engine: ${engine_path}"
fi

python tools/validate_phase12_tensorrt.py \
  --config "${config}" \
  --checkpoint "${checkpoints[0]}" \
  --onnx "${ort_onnx_path}" \
  --engine "${engine_path}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/test \
  --samples 20 \
  --warmup 20 \
  --iters 100 \
  --input-size 512 \
  "$@"
