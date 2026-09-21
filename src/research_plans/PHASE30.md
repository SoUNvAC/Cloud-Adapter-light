# Phase 30 — Final frozen evidence audit

## Purpose

Phase 29 failed two hard gates, so Phase 30 is not another model search. It is a
read-only audit that freezes the Phase 21-29 evidence chain, verifies statistical
and protocol consistency, and records the exact claims that the current results
do and do not support. No dataset inference or training is run, and the internal
CloudSEN test remains sealed.

## Pre-registered audit gates

- Phase 21 data-integrity audit and Phase 22 clean V8 baseline must pass.
- Observed phase decisions must match their immutable summaries: Phase 23/24/25
  fail, Phase 26/27 pass, and Phase 28/29 fail.
- Phase 22 and Phase 27 means/sample standard deviations must recompute from
  their run records within numerical tolerance.
- Every checkpoint referenced by the two three-seed summaries must exist.
- All post-Phase-21 summaries must record `test_evaluated=false`; Phase 28/29
  must additionally record `internal_test_evaluated=false`.
- Benchmarks compared in the efficiency claims must use batch 1, 512x512,
  native PyTorch FP16 with resident FP16 weights.
- The tracked remote worktree must be clean and the repository test suite must
  pass. Untracked generated checkpoints and work directories are not modified.

Audit success means only that the evidence is internally consistent. It does
not convert a failed model gate into a pass. The final report must state that V8
is the clean accuracy anchor, ResNet-18 is an internal efficiency point rejected
by the external gate, MobileNetV2 is rejected, and no compressed successor has
qualified for internal-test unsealing.

## One-click command

```bash
cd src
bash tools/run_phase30_final_audit_oneclick_4090d.sh
```

The single terminal report is
`work_dirs/phase30_final_audit/PHASE30_REPORT.txt`, with machine-readable
`summary.json` and `FINAL_EVIDENCE_REPORT.md` beside it.
