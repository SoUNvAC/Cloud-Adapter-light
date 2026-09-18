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
