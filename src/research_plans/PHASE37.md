# Phase 37 — weak-class supervised reweighting pilot

## Motivation

Phase 31's deployment graph remains the only compact graph above 2x speedup.
Phases 34-36 show that teacher KD, extra decoder fusion, and a larger mobile
backbone do not meet the joint gates. Thin cloud and shadow remain the dominant
accuracy gap, so this phase changes training supervision without adding any
inference operation.

## Fixed protocol

- Architecture, ImageNet initialization, optimizer, schedule, seed, and
  40,000-iteration length are identical to Phase 31.
- The only change is Mask2Former query-classification weights:
  clear/thick/thin/shadow/no-object = `[1.0, 1.0, 1.5, 1.25, 0.1]`.
- Mask BCE, Dice loss, Hungarian costs, point sampling, and all other settings
  remain unchanged; there is no teacher or auxiliary deployment branch.
- Independent seed-42 training. Selection uses official CloudSEN validation;
  L8 and internal test are not read.

## Pre-registered gates and stop-loss

- validation mIoU >=68.0;
- mean IoU of thin cloud and cloud shadow >=51.0;
- native-FP16 speedup versus V8 >=2.0x;
- deployment parameters <=3.0M;
- all metrics finite and internal test remains sealed.

If every gate passes, Phase 38 freezes the checkpoint for one L8 zero-shot
evaluation. If any gate fails, supervised class reweighting is closed without
changing weights, schedule, seed, or thresholds.

## One-click command

```bash
cd src
bash tools/run_phase37_weakclass_supervision_oneclick_4090d.sh
```

The report is `work_dirs/phase37_weakclass_supervision/PHASE37_REPORT.txt`.
