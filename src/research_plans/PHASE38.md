# Phase 38 — training-only boundary semantic supervision

## Motivation

Phase 37 misses the global gate by 0.65 but passes the weak-class and 2x speed
gates, showing that ground-truth supervision helps whereas teacher KD does not.
This phase restores the original uniform query-classification loss and tests a
distinct training-only pixel-semantic objective concentrated on weak classes
and label boundaries.

## Fixed protocol

- Inference architecture and all base Mask2Former losses are exactly Phase 31;
  query classification returns to `[1, 1, 1, 1, 0.1]`.
- Add semantic-logit cross entropy during training only, with class weights
  `[1.0, 1.0, 1.5, 1.25]`, boundary multiplier 1.5, radius 1, and total weight
  0.5. The auxiliary objective adds no checkpoint parameter or inference op.
- Independent seed-42 training from the same ImageNet initialization for
  40,000 iterations. No Phase 37 warm start and no teacher.
- Selection uses official CloudSEN validation. L8 and internal test are unread.

## Pre-registered gates and stop-loss

- validation mIoU >=68.0;
- mean IoU of thin cloud and cloud shadow >=51.0;
- native-FP16 speedup versus V8 >=2.0x;
- deployment parameters <=3.0M;
- all metrics finite and internal test remains sealed.

If every gate passes, Phase 39 performs one frozen L8 zero-shot evaluation. If
any gate fails, supervision-loss variants are closed without changing weights,
radius, duration, seed, or thresholds.

## One-click command

```bash
cd src
bash tools/run_phase38_boundary_supervision_oneclick_4090d.sh
```

The report is `work_dirs/phase38_boundary_supervision/PHASE38_REPORT.txt`.
