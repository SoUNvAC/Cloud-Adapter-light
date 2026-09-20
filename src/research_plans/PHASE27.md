# Phase 27 — Three-seed confirmation of the ResNet-18 compact model

## Dependency

Phase 26 passed all pre-registered accuracy, speed, and size gates. Phase 27
therefore freezes its architecture, ImageNet initialization, optimizer,
learning-rate multipliers, 40,000-iteration schedule, and official validation
protocol. The completed seed-42 run is reused without retraining; only seeds
123 and 3407 are added.

## Fixed protocol

- Configuration: `configs/protocol/phase26_resnet18_v8_l1c.py` unchanged.
- Seeds: 42 (Phase 26), 123, and 3407.
- Each new run starts independently from the same verified official OpenMMLab
  ImageNet checkpoint and selects its best checkpoint only on official val.
- Test stays sealed. No learning-rate, schedule, loss, or threshold repair is
  allowed after observing the additional seeds.
- Deployment evidence is inherited from the frozen Phase 26 graph: 12.439M
  parameters and 1.828x native-FP16 speedup in the paired 4090D measurement.

## Pre-registered gates and stop-loss

- Every seed must reach at least 67.0 validation mIoU.
- Three-seed mean validation mIoU must reach at least 67.8.
- Three-seed sample standard deviation must not exceed 0.60 mIoU.
- The frozen Phase 26 deployment gates must remain passed: speedup >=1.25x and
  parameters <=18.0M.

If all gates pass, Phase 28 proceeds to a genuinely unseen geographic holdout
or external cloud dataset before unsealing the internal test. If any gate
fails, the ResNet-18 direction is closed and Phase 28 switches to a separately
pretrained mobile backbone; thresholds are not relaxed.

## One-click command

```bash
cd src
bash tools/run_phase27_resnet18_3seed_oneclick_4090d.sh
```

The single report is
`work_dirs/phase27_resnet18_3seed/PHASE27_REPORT.txt`. The script never starts
Phase 28.
