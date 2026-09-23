# Phase 53 — cloud-shadow auxiliary supervision diagnostic

## Frozen Phase 52 diagnosis

All comparisons use the same 1,905-image scene-disjoint target-val split.

| Metric | Source-only | Phase 52A | Delta |
|---|---:|---:|---:|
| clear IoU | 74.7081 | 79.2943 | +4.5862 |
| thick-cloud IoU | 67.1766 | 69.6455 | +2.4689 |
| thin-cloud IoU | 14.1403 | 29.6299 | +15.4896 |
| cloud-shadow IoU | 16.4714 | 0.8403 | -15.6312 |
| thin-cloud Boundary F1 | 2.8838 | 20.9335 | +18.0497 |
| cloud-shadow Boundary F1 | 17.0990 | 0.5012 | -16.5978 |

The Phase 52 checkpoint with its target path disabled scores 73.9800 source-val
mIoU, exactly matching the frozen source baseline at report precision. Thus
the architecture provides source zero-forgetting by sensor-path selection.

The improvement is not confined to clear/thick cloud: sparse MsRE strongly
improves thin-cloud regions and boundaries. The failure is specifically a
cloud-shadow collapse. This rejects the broad claim that MsRE only corrects
generic scale shift; it supports the narrower hypothesis that scale refinement
helps translucent cloud while the rare shadow semantic is overwhelmed.

## Single changed mechanism

Phase 53 retains Phase 52's 16 generic tokens, 380,577 trainable parameters,
injection blocks `[2, 5, 8, 11]`, rank-8 head delta, frozen source path,
preselected 65 target patches, seed 52, optimizer, augmentations, learning rate,
validation cadence and 4,000 iterations. It adds only a parameter-free binary
cloud-shadow auxiliary loss on semantic logits. Positive and negative pixels
are averaged separately and then given equal mass; the fixed auxiliary weight
is 0.5. No ordinal, boundary, sampling, pseudo-label, metadata, or class-token
mechanism is included.

## Stop-loss

The selected checkpoint must satisfy every gate:

- target mIoU at least 46.1241 (+3.0 over source-only);
- weak-class mIoU at least 19.3059 (+4.0 over source-only);
- cloud-shadow IoU at least 16.4714 (recover the source-only level);
- cloud-shadow Boundary F1 at least 10.0;
- thin-cloud IoU at least 27.6299 (no more than 2.0 below Phase 52A);
- source-val mIoU at least 73.93 with the target path disabled;
- exactly 380,577 target-trainable parameters and no inference-time operator
  change relative to Phase 52A;
- complete finite evaluation, with target-test and CloudSEN internal test sealed.

Any failed gate stops this mechanism. The loss weight, sampler, token count,
injection position, schedule and selection must not be adjusted afterward.
