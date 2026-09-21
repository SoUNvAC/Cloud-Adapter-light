# Phase 33 — LiteFPN three-seed clean-validation stability

## Motivation

Phase 31 passed clean validation, parameter, and native-FP16 latency gates, and
Phase 32 passed the frozen L8 zero-shot gate by only 0.01 mIoU. Before any
further claim, the training result must be shown to be reproducible rather than
a favorable seed.

## Fixed protocol

- Reuse the completed Phase 31 seed 42 run unchanged.
- Independently train seeds 123 and 3407 for 40,000 iterations with the exact
  Phase 31 architecture, ImageNet initialization, optimizer, and schedule.
- Select checkpoints only on official CloudSEN validation.
- Do not read L8 in this phase and keep CloudSEN internal test sealed.
- Reuse the frozen Phase 31 deployment measurements because the graph is
  identical: 2.548229M parameters and 2.050585x native-FP16 speedup.

## Pre-registered gates and stop-loss

- every seed validation mIoU >=64.5;
- three-seed mean validation mIoU >=65.3;
- sample standard deviation <=0.75;
- frozen speedup >=2.0x and parameters <=3.0M;
- all metrics finite and internal test remains sealed.

If every gate passes, the next phase may quantify cross-seed L8 variance using
the same frozen protocol. If any gate fails, the candidate is not a stable
success and no threshold, seed set, or training hyperparameter is changed.

## One-click command

```bash
cd src
bash tools/run_phase33_litefpn_3seed_oneclick_4090d.sh
```

The report is `work_dirs/phase33_litefpn_3seed/PHASE33_REPORT.txt`.
