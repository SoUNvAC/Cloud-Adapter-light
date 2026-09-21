# Phase 32 — frozen LiteFPN external zero-shot gate

## Motivation

Phase 31 is the first compact successor to pass the pre-registered clean-val,
parameter, and native-FP16 latency gates. This phase tests whether that fixed
checkpoint retains the existing cross-sensor floor before any extra seeds or
internal-test access are allowed.

## Fixed protocol

- Freeze the Phase 31 seed-42 best checkpoint without training, calibration,
  threshold search, test-time augmentation, or checkpoint reselection.
- Evaluate once on all 2,643 Landsat-8 Biome test images at 512x512.
- Reuse the pre-registered CloudSEN-to-L8 label remap `[0, 3, 2, 1]`.
- Compare with the frozen Phase 28 V8 three-seed mean of 39.7533 mIoU.
- CloudSEN internal test remains sealed.

## Pre-registered gates and stop-loss

- L8 zero-shot mIoU >=35.0;
- drop from the Phase 28 V8 mean <=6.5 points;
- all aggregate metrics finite;
- exactly 2,643 images evaluated and internal test remains sealed.

If every gate passes, Phase 33 keeps the architecture and hyperparameters fixed
and trains seeds 123 and 3407 for clean-validation stability. If any gate fails,
the MobileNetV2 + LiteFPN direction is closed without L8-driven tuning or a
changed class mapping.

## One-click command

```bash
cd src
bash tools/run_phase32_l8_litefpn_oneclick_4090d.sh
```

The report is `work_dirs/phase32_l8_litefpn/PHASE32_REPORT.txt`.
