# Phase 22 — Clean-protocol V8 statistical baseline

## Research question

Can the V8 architecture reproduce a stable high-accuracy baseline when checkpoint
selection is restricted to the official 535-image validation partition?

Historical Phase 8 used the 975-image test partition as validation. Its 71.28 mIoU
is retained as development history but is not a valid acceptance target for the
new protocol.

## Training

Run on the 24-GiB RTX 4090D environment:

```bash
cd src
bash tools/run_phase22_oneclick_4090d.sh
```

The one-click entry prepares missing DINOv2-S weights, locates either supported
dataset layout, runs all three seeds, validates and summarizes them, and writes
the complete console transcript plus final JSON to one file:
`work_dirs/phase22_clean_v8/PHASE22_REPORT.txt`. Send only that report for the
Phase 22 decision. It never starts Phase 23. Interrupted runs with an MMEngine
`last_checkpoint` resume automatically; completed seeds are reused.

This independently trains seeds 42, 123, and 3407 for 40,000 iterations. Each
run starts from the same converted DINOv2-S initialization. It does not load a
Phase 1--20 student checkpoint. The official validation split selects the best
checkpoint. The test split is not evaluated in Phase 22.

## Pre-registered gates

- Every run must finish and produce exactly one `best_mIoU_iter_*.pth`.
- Every best checkpoint must achieve validation mIoU >= 69.0.
- Three-seed mean validation mIoU must be >= 69.5.
- Sample standard deviation of validation mIoU must be <= 0.50.

## Stop-loss decision

If any gate fails, do not start a novel compression method and do not relax the
threshold after seeing the results. First run a training-stability diagnosis:
verify checkpoint initialization, inspect per-seed loss/gradient traces, and
compare deterministic versus benchmark CUDA kernels on validation only. If the
mean remains below 69.5 after fixing an implementation defect, V8 is rejected as
the clean high-accuracy anchor and Phase 23 switches to a freshly trained V7 or
Control anchor. Test remains sealed until the method and all thresholds are
frozen.
