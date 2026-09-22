# Phase 40 — frozen compact-model validation error stratification

## Motivation

Phase 39 found no paper-qualified compact candidate and closed five adjacent
trial-and-error directions.  Phase 37 is nevertheless the strongest compact
operating point (67.35 validation mIoU, 51.555 weak-class mIoU, 2.016x speedup).
Before proposing another trainable method, this phase diagnoses where its thin
cloud and cloud-shadow errors occur.

## Fixed protocol

- Freeze the Phase 37 seed-42 best checkpoint and inference graph.
- Run exactly once on all 535 official CloudSEN validation images at 512x512
  with FP16 weights.  Do not read L8 or CloudSEN internal test.
- Accumulate the ordinary four-class confusion matrix and separately measure
  weak-class recall on one-pixel label boundaries and interiors.
- Sort images deterministically by ground-truth weak-class pixel prevalence
  (filename breaks ties), split into four near-equal groups, and report
  aggregate thin/shadow IoU for every quartile.  These strata use labels only;
  no checkpoint or threshold is selected from them.
- Save a single machine summary plus per-image CSV for reproducibility.

## Pre-registered integrity gates and decision rule

- exactly 535 unique image/annotation pairs are evaluated;
- recomputed aggregate mIoU is within 0.05 point of the frozen 67.35 value;
- every boundary/interior and prevalence stratum has valid pixels and finite
  statistics;
- internal test remains sealed.

The diagnostic direction is selected without changing thresholds:

1. if weak-class interior recall minus boundary recall is at least 8.0 points,
   select a high-resolution boundary-preserving architectural method;
2. otherwise, if highest-prevalence minus lowest-prevalence weak mIoU is at
   least 8.0 points, select training-data rarity-aware sampling;
3. otherwise select a context-modeling method.

If an integrity gate fails, fix only the evaluator and rerun the frozen audit;
do not train.  If it passes, Phase 41 may implement exactly the selected method
and must retain the 68.0/51.0/2.0x/3.0M paper-level stop lines.

## One-click command

```bash
cd src
bash tools/run_phase40_error_audit_oneclick_4090d.sh
```

The report is `work_dirs/phase40_error_audit/PHASE40_REPORT.txt`.
