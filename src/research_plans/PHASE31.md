# Phase 31 — MobileNetV2 plus standard-operator LiteFPN pilot

## Motivation

Phase 29 reduced the full model to 2.992M parameters but achieved only 1.609x
speedup, showing that the two-layer deformable pixel decoder now dominates
latency. Earlier LightCloudHead variants were fast but had a confirmed accuracy
bottleneck, so Phase 31 does not repeat them. It retains Mask2Former query
reasoning and replaces only the deformable pixel decoder with the existing
standard-operator depthwise-separable `LiteFPNPixelDecoder`.

## Fixed protocol

- Backbone and ImageNet initialization: Phase 29 MobileNetV2, width 1.0.
- Decoder: Phase 29 128-channel, 25-query, two-layer query decoder; only the
  pixel decoder is replaced by LiteFPN.
- Seed 42, independent 40,000-iteration training from ImageNet initialization;
  no Phase 29 segmentation weights are transferred.
- Checkpoint selection uses only official CloudSEN validation. Internal test is
  sealed. L8 is not read in this phase, so it cannot tune the speed screen.
- Native PyTorch FP16 benchmark: batch 1, 512x512, 20 warmups, 100 iterations,
  compared with the frozen Phase 26 V8 benchmark anchor.

## Pre-registered gates and stop-loss

- validation mIoU >=65.5;
- native-FP16 speedup versus V8 >=2.0x;
- deployment parameters <=3.0M;
- all metrics and latencies finite and positive.

If all gates pass, Phase 32 freezes the model and performs one zero-shot L8
evaluation under the existing taxonomy mapping. If any gate fails, this decoder
replacement is closed without learning-rate or width repair, and the next
direction must remove Mask2Former query decoding entirely or use a deployment
backend rather than further backbone compression.

## One-click command

```bash
cd src
bash tools/run_phase31_mobilenetv2_litefpn_oneclick_4090d.sh
```

The report is
`work_dirs/phase31_mobilenetv2_litefpn/PHASE31_REPORT.txt`. No later phase is
started automatically.
