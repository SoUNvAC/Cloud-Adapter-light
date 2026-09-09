#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

source tools/activate_onnxruntime_cuda_libs.sh

config="configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py"
checkpoint_dir="work_dirs/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c"
onnx_path="work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx"
output_dir="work_dirs/phase16_v12_tensorrt"
engine_path="${output_dir}/v12_export_fpn_fp16_b1_512.engine"
checkpoints=("${checkpoint_dir}"/best_mIoU_iter_*.pth)

if ((${#checkpoints[@]} != 1)); then
  echo "Expected one Phase 12 best checkpoint in ${checkpoint_dir}, found ${#checkpoints[@]}"
  exit 1
fi
if [[ ! -f "${onnx_path}" ]]; then
  echo "Missing ${onnx_path}; run tools/export_phase14_onnx_4090d.sh first"
  exit 1
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
if [[ "${REBUILD_ENGINE:-0}" == "1" ]]; then
  python tools/build_phase12_tensorrt.py \
    --onnx "${onnx_path}" \
    --output "${engine_path}" \
    --workspace-gib 4 \
    --force
elif [[ ! -f "${engine_path}" ]]; then
  python tools/build_phase12_tensorrt.py \
    --onnx "${onnx_path}" \
    --output "${engine_path}" \
    --workspace-gib 4
else
  echo "Reusing existing engine: ${engine_path}"
fi

python tools/validate_phase12_tensorrt.py \
  --config "${config}" \
  --checkpoint "${checkpoints[0]}" \
  --onnx "${onnx_path}" \
  --engine "${engine_path}" \
  --image-dir data/cloudsen12_high_l1c/img_dir/test \
  --samples 20 \
  --warmup 20 \
  --iters 100 \
  --input-size 512 \
  "$@"
