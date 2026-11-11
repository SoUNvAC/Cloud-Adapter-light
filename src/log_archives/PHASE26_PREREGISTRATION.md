# Phase 26 preregistration: real INT8 PTQ

Phase 26 starts only after exactly one Phase 25 candidate passes all paired
three-seed validation gates. The official test split remains sealed.

## Frozen selection and calibration

- Select the median validation-mIoU Phase 25 run, with seed as the exact-tie
  breaker. Do not select the best seed after inspecting deployment results.
- Export the selected statically pruned network as FP32 ONNX at 512 x 512.
- Calibrate explicit Q/DQ INT8 with 256 deterministic, evenly spaced images
  from the official 535-image validation split. Test images are forbidden.
- Quantize only Conv, Gemm, and MatMul. Keep normalization, softmax, resize,
  sigmoid, and the final semantic reduction outside INT8.
- Build TensorRT FP16 and TensorRT INT8-Q/DQ engines from the same checkpoint,
  block subset, input contract, TensorRT version, workspace, and GPU.

Run on the pinned 4090D environment:

```bash
bash tools/run_phase26_real_int8_4090d.sh
```

The runner resolves the Phase 25 checkpoint, exports and calibrates the model,
builds both engines, evaluates all 535 validation images, and writes
`work_dirs/phase26_real_int8/summary.json`.

## Hard gates and stop-loss

All gates must pass simultaneously:

1. FP32 ONNX validation mIoU differs from the selected Phase 25 PyTorch result
   by no more than 0.05 percentage point.
2. Q/DQ ONNX and TensorRT INT8 each lose no more than 0.50 mIoU relative to
   FP32 ONNX; TensorRT FP16 differs by no more than 0.05.
3. Q/DQ ONNX and TensorRT INT8 each retain at least 98.50% pixel agreement with
   FP32 ONNX, all logits are finite, and no core compute operator falls back to
   the CPU in ONNX Runtime profiling.
4. TensorRT INT8 mean kernel latency is at least 1.15x faster than TensorRT
   FP16. The benchmark rotates 20 validation images, uses batch 1, 20 warmups,
   and 200 timed iterations with CUDA synchronization. Inputs and outputs stay
   on the GPU for both engines.

If any gate fails, the PTQ INT8 direction is closed. Fake quantization or the
mere presence of an INT8 builder flag must not be reported as real acceleration.
The next direction is FP16 structured channel/token reduction, followed by the
same full-validation accuracy and same-backend latency gates. Phase 26 is added
to `EXPERIMENT_LOG.md` only after measured completion, whether pass or fail.
