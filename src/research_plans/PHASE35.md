# Phase 35 — bidirectional LiteFPN architecture pilot

## Motivation

Phase 34 confirms that semantic KD does not repair the structurally mismatched
compact student. Phase 31's one-way top-down LiteFPN gives the finest mask
feature semantic context, but the coarser features consumed by query attention
do not receive refined fine detail. Phase 35 adds a low-cost bottom-up pass only
at stride 8 and coarser levels.

## Fixed protocol

- MobileNetV2 and the 128-channel, 25-query, two-layer query decoder remain
  unchanged from Phase 31.
- Replace LiteFPN with BiLiteFPN: retain its top-down path, then bilinearly
  downsample refined detail into each coarser query level using normalized
  non-negative two-input weights and one depthwise-separable refinement block.
- All deployment operations remain standard Conv/GN/ReLU/Resize/Add operations.
- Independent seed-42 training from the same ImageNet initialization for 40,000
  iterations. No Phase 31 or Phase 34 segmentation weights and no teacher.
- Selection uses only official CloudSEN validation. L8 and internal test remain
  unread.

## Pre-registered gates and stop-loss

- validation mIoU >=68.0;
- mean IoU of thin cloud and cloud shadow >=51.0;
- native-FP16 speedup versus V8 >=2.0x;
- deployment parameters <=3.0M;
- all metrics finite and internal test remains sealed.

If every gate passes, Phase 36 freezes the checkpoint for one L8 zero-shot
evaluation. If any gate fails, bidirectional LiteFPN is closed without changing
fusion weights, channel width, training schedule, or thresholds.

## One-click command

```bash
cd src
bash tools/run_phase35_bilitefpn_oneclick_4090d.sh
```

The report is `work_dirs/phase35_bilitefpn/PHASE35_REPORT.txt`.
