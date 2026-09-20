# Phase 23 — Clean-protocol V12 paired baseline

## Dependency

Phase 23 may start only when `work_dirs/phase22_clean_v8/summary.json` reports
`passed: true` and `test_evaluated: false`. Every V12 run is warm-started from
the clean V8 checkpoint trained with the same random seed.

## Research question

Does the standard-operator V12 decoder preserve a stable validation Pareto
anchor after the Phase 1--20 test-selection leakage is removed?

## Training

```bash
cd src
bash tools/run_phase23_clean_v12_3seed_4090d.sh
```

For unattended execution with a single terminal artifact, use:

```bash
bash tools/run_phase23_oneclick_4090d.sh
```

It captures setup, training, validation, the final gate decision, and
`summary.json` in `work_dirs/phase23_clean_v12/PHASE23_REPORT.txt`. It resumes
an interrupted run from MMEngine's `last_checkpoint`, retries an interrupted
validation, and never starts Phase 24.

The script runs seeds 42, 123, and 3407 for 20,000 iterations. Checkpoints are
selected and re-evaluated only on the official validation split. Test remains
sealed.

## Pre-registered gates

- Phase 22 must have passed without test evaluation.
- Every run must produce exactly one best checkpoint.
- Every V12 validation mIoU must be >= 66.5.
- Three-seed mean validation mIoU must be >= 67.0.
- Sample standard deviation must be <= 0.60.
- Mean paired V8-to-V12 mIoU drop must be <= 3.0 points.

## Stop-loss decision

If accuracy or variance fails, V12 is rejected as the clean deployment anchor.
The next direction is not threshold relaxation: first test whether the FPN
warm-start and learning-rate multiplier cause the instability. At most one
pre-registered repair is allowed, replacing the fixed 5x FPN learning-rate
multiplier with a short linear sweep on validation. If that repair still fails,
the standard-FPN branch is closed and the project moves to structured backbone
compression while retaining the V8 deformable decoder as the accuracy anchor.

## Repair A preregistration (written after Trial A, before repair results)

Trial A passed the absolute accuracy and variance gates but missed the paired
drop gate: 70.5533 mean mIoU, 0.1320 sample standard deviation, and 3.2500 mean
paired drop versus the 3.0 limit. The threshold remains unchanged.

The one allowed repair is a short seed-42 sweep of FPN learning-rate
multipliers `[1, 2, 3, 4]`. Each candidate starts from the same Phase 22 seed-42
checkpoint, trains for 8,000 iterations, and selects checkpoints only on the
official validation split. The observed Trial-A 5x seed-42 best result within
its first 8,000 iterations was 67.58 mIoU. A repair candidate qualifies only if
its best 8k validation mIoU is at least 67.83, a predeclared +0.25 margin. The
highest qualifying result is selected; an exact tie chooses the lower
multiplier. Test remains sealed.

If no candidate qualifies, Phase 23 closes as failed immediately. Otherwise the
selected multiplier is frozen and rerun for the original 20,000 iterations on
all three paired seeds. The original gates are reused without relaxation:
every run >=66.5 mIoU, mean >=67.0, sample std <=0.60, and mean paired drop
<=3.0. A second miss closes the standard-FPN branch; there is no further LR,
schedule, seed, or threshold search.
