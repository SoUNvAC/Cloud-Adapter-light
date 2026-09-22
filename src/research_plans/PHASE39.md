# Phase 39 — frozen Phase 31–38 Pareto and evidence audit

## Motivation

Phase 38 closes the pre-registered supervision-loss direction: it misses both
the 68.0 global validation gate and the 51.0 weak-class gate.  Starting another
nearby loss or backbone trial without first consolidating the evidence would be
architecture roulette.  This phase therefore performs no training and no model
selection; it freezes the Phase 31–38 evidence and determines whether any
candidate is eligible for the next external or internal-test gate.

## Fixed protocol

- Read only the existing Phase 31–38 machine-readable summaries.
- Do not run inference and do not read CloudSEN internal test.
- Recompute Phase 33 mean/sample standard deviation and verify every recorded
  pass/fail decision.
- Apply the paper-level compact-candidate gates uniformly: validation mIoU
  >=68.0, weak-class mIoU >=51.0, native-FP16 speedup >=2.0x, and parameters
  <=3.0M.  Stability additionally requires three-seed standard deviation
  <=0.75; external eligibility requires the frozen Phase 32 L8 gates.
- The historical Phase-specific pilot passes remain recorded but do not count
  as paper-level qualification.

## Pre-registered gates and stop-loss

The audit itself passes only if all eight summaries exist, decisions and Phase
33 statistics reproduce exactly, benchmark protocol fields match, and internal
test is sealed throughout.  A model is qualified only if the uniform accuracy,
weak-class, efficiency, stability, and external gates are all supported by the
existing evidence.  Thresholds will not be relaxed after inspection.

If no candidate qualifies, Phase 40 must be a frozen, stratified error audit of
the strongest compact checkpoint before any new trainable method is proposed.
No further class-weight, auxiliary-loss, MobileNet-family, LiteFPN-fusion, or
teacher-KD variants are allowed.

## One-click command

```bash
cd src
bash tools/run_phase39_pareto_audit_oneclick_4090d.sh
```

The report is `work_dirs/phase39_pareto_audit/PHASE39_REPORT.txt`.
