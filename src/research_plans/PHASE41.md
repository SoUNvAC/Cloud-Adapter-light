# Phase 41 — stride-4 high-frequency detail residual

## Motivation

The frozen Phase 40 audit reproduced Phase 37 and measured a 22.7601-point
weak-class recall deficit at label boundaries versus interiors, exceeding the
pre-registered 8.0-point diagnostic threshold.  This phase tests the selected
high-resolution architectural direction without reopening failed boundary-loss
or class-weight searches.

## Fixed method

- Use the complete Phase 37 training recipe, including its already frozen
  query-class weights; the only experimental change is the pixel decoder.
- Keep the ordinary LiteFPN top-down and query paths unchanged.
- From the raw stride-4 backbone feature, apply a 24-to-16 pointwise projection,
  subtract a 3x3 average-pooled local mean, project the high-frequency residual
  to 128 channels, and add it only to the stride-4 mask feature with one learned
  scalar initialized to 0.1.  Query multi-scale features are untouched.
- Train independently from the same ImageNet initialization for 40,000
  iterations with seed 42.  Do not warm-start Phase 37.
- Select only on official CloudSEN validation; do not read L8 or internal test.

## Pre-registered gates and stop-loss

- validation mIoU >=68.0;
- mean thin-cloud/cloud-shadow IoU >=51.0;
- native-PyTorch FP16 speedup versus frozen V8 >=2.0x;
- deployment parameters <=3.0M;
- all metrics finite and internal test sealed.

If any gate fails, close this high-frequency detail branch without changing its
width, kernel, gate initialization, training duration, seed, or thresholds.  If
all pass, the next phase performs one frozen L8 zero-shot evaluation before any
additional CloudSEN seed or internal-test access.

## Expected outcome

The experiment is successful only if it recovers at least 0.65 validation point
over Phase 37 while retaining its weak-class and real 2x deployment gates.  A
visual boundary improvement or a sub-68 result is not a pass.
