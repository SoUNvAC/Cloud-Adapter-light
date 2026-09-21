# Phase 36 — MobileNetV3-Large plus LiteFPN pilot

## Motivation

Phase 35 shows that adding decoder fusion costs the entire latency margin while
recovering only 0.78 mIoU. The next independent variable is therefore backbone
representation efficiency. MobileNetV3-Large supplies squeeze-excitation and
hard-swish blocks with independent ImageNet pretraining while retaining a
mobile deployment graph.

## Fixed protocol

- Backbone: torchvision MobileNetV3-Large IMAGENET1K_V2 weights, with outputs
  after feature blocks 3/6/12/15 (strides 4/8/16/32; channels 24/40/112/160).
- Decoder: the original one-way Phase 31 LiteFPN, 128 channels, 25 queries and
  two query-decoder layers. No bidirectional pass and no distillation.
- Seed 42, independent 40,000-iteration training; backbone learning rate is
  0.1x the decoder rate as in the existing compact-backbone protocol.
- Checkpoint selection uses only official CloudSEN validation. L8 and internal
  test are not read.

## Pre-registered gates and stop-loss

- validation mIoU >=68.0;
- mean IoU of thin cloud and cloud shadow >=51.0;
- native-FP16 speedup versus V8 >=2.0x;
- deployment parameters <=4.0M;
- all metrics finite and internal test remains sealed.

If every gate passes, Phase 37 freezes the checkpoint for one L8 zero-shot
evaluation. If any gate fails, this backbone direction is closed without
changing width, output blocks, decoder, schedule, or thresholds.

## One-click command

```bash
cd src
bash tools/run_phase36_mobilenetv3_litefpn_oneclick_4090d.sh
```

The report is `work_dirs/phase36_mobilenetv3_litefpn/PHASE36_REPORT.txt`.
