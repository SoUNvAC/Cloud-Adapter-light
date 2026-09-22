# Phase 43 — global scene-context gated LiteFPN

## Motivation

Phase 42 failed and closes prevalence-balanced sampling.  Under Phase 40's
pre-registered decision tree, context modeling is now the only remaining method
direction.  This phase tests whether global scene context can disambiguate thin
cloud and cloud shadow without sacrificing the 2x deployment budget.

## Fixed method

- Keep every Phase 37 setting, including MobileNetV2, LiteFPN width, query
  decoder, fixed class weights, initialization, optimizer, schedule, seed 42,
  augmentations, and 40,000 iterations.
- Global-average-pool the raw stride-32 320-channel backbone feature and apply
  one learned 1x1 projection to 128 channels.
- Form a channel gate as `2 * sigmoid(projected_context)`, so zero projection
  produces an identity gate.  Initialize projection weights and bias to zero.
- Multiply all four refined LiteFPN feature levels by the same scene-conditioned
  gate before both query-feature selection and mask-feature projection.
- No extra spatial convolution, loss, sampler, teacher, or test-time method.
- Select on official CloudSEN validation only; L8 and internal test remain unread.

## Pre-registered gates and stop-loss

- validation mIoU >=68.0;
- mean thin-cloud/cloud-shadow IoU >=51.0;
- native-PyTorch FP16 speedup versus frozen V8 >=2.0x;
- deployment parameters <=3.0M;
- all metrics finite and internal test sealed.

If any gate fails, close compact context gating without modifying its source
level, projection width, initialization, placement, duration, seed, or
thresholds.  No further adjacent compact-model trial is permitted; the next
phase must freeze and report the negative Phase 31–43 evidence.  If all gates
pass, perform one frozen L8 zero-shot evaluation next.

## Expected outcome

The method must improve Phase 37 by at least 0.65 validation point while
retaining weak-class mIoU and real 2x speed.  Parameter or latency eligibility
without 68.0 mIoU remains a failure.
