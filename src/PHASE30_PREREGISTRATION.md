# Phase 30 preregistration: Jetson Orin Nano target deployment

Phase 30 is the first target-device claim. Desktop 4090D latency, an engine
built on another GPU, simulated quantization, or a PyTorch-backed runner cannot
substitute for this phase.

## Frozen target and artifact contract

- Device: NVIDIA Jetson Orin Nano 8GB, JetPack 6.1 or later in the JetPack 6
  family, aarch64, CUDA 12, TensorRT 10, the image's advertised MAXN or
  MAXN_SUPER power mode, and clocks locked with `jetson_clocks`.
- Power-mode IDs are not portable across ordinary and Super configurations.
  Inspect `sudo nvpmodel -q --verbose`, select the ID explicitly labelled
  MAXN or MAXN_SUPER with `sudo nvpmodel -m <ID>`, and only then run
  `sudo jetson_clocks`. The audit records full query output and rejects a
  non-MAXN or unlocked CPU/GPU/EMC run.
- Transfer only the selected Phase 25/26 ONNX and selection JSON to the Jetson.
  TensorRT engines must be rebuilt locally and record the target device model.
- Append ONNX ArgMax and uint8 Cast before engine construction. Runtime I/O is
  `float32[1,3,512,512]` RGB 0..255 to `uint8[1,512,512]` semantic mask.
- Runtime dependencies are NumPy, Pillow, TensorRT, and cuda-python. Importing
  PyTorch is a hard failure.

Install Python-side dependencies without replacing JetPack TensorRT:

```bash
python3 -m pip install -r requirements-jetson.txt
sudo nvpmodel -q --verbose
sudo nvpmodel -m <MAXN_ID_FROM_THE_PREVIOUS_COMMAND>
sudo jetson_clocks
bash tools/run_phase30_jetson_orin_nano.sh mixed-fp16
```

If and only if Phase 26 INT8 passes on 4090D and its passing `summary.json`
and Q/DQ ONNX are transferred, also run:

```bash
bash tools/run_phase30_jetson_orin_nano.sh int8-qdq
```

## Accuracy, timing, and power protocol

- Accuracy: all 535 official validation images; CloudSEN12 test remains sealed.
- Accuracy reference: the median-seed Phase 25 validation mIoU frozen in
  `selection.json`.
- Timing contract: 20 decoded/preloaded CPU float32 images rotated through
  synchronous H2D, TensorRT execution, D2H, and returned CPU mask. Image file
  decoding is excluded. Batch size is one.
- Run 50 warmups followed by five independent repeats of 200 timed images.
  Report first, mean, P50, P90, P99, throughput, each repeat mean, and repeat CV.
- Sample `tegrastats` every 100ms. With the engine resident, first measure a
  10-second idle window, then the warmup and timed window. Report idle/active
  VDD_IN, peak power, RAM, temperature, total energy/image, and idle-subtracted
  dynamic energy/image.

## Hard gates and stop-loss

All gates must pass:

1. Full validation mIoU drop from Phase 25 <= 0.50 point; all masks contain only
   classes 0..3.
2. Mean end-to-end latency <= 80ms and P90 <= 100ms (at least 10 img/s at P90).
3. Repeat-mean coefficient of variation <= 5%.
4. Total energy <= 2.0J/image, peak VDD_IN <= 25W, peak temperature <= 80C,
   and peak system RAM <= 7,500MB.
5. The environment audit, target-local engine provenance, anti-lazy checksum,
   and no-PyTorch runtime checks all pass.

If the ONNX mask graph cannot build, that is a failed deployment contract; a
logits engine with a 4MB host transfer cannot be silently substituted. If any
gate fails, do not claim target real-time deployment. Switch to FP16
whole-network channel/token reduction and repeat this unchanged protocol.
Phase 30 is appended to `EXPERIMENT_LOG.md` only after the physical-device run.
