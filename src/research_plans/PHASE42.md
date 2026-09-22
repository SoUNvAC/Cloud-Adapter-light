# Phase 42 — training-only weak-prevalence balanced sampling

## Motivation

Phase 41 failed the global gate and closes the high-frequency detail branch.
Phase 40's second pre-registered diagnostic was a 37.3677-point weak-class IoU
gap between the highest and lowest ground-truth weak-prevalence quartiles.  This
phase tests data rarity directly while preserving the Phase 37 deployment graph.

## Fixed method

- Keep the full Phase 37 architecture, initialization, losses, optimizer,
  schedule, augmentations, seed 42, and 40,000-iteration duration unchanged.
- Before training, rank all 8,490 training images by the fraction of valid
  pixels labeled thin cloud or cloud shadow; path is the deterministic tie
  breaker.  Define the highest-ranked 25% as the weak-rich pool.
- Each sampler draw has a fixed 50% probability of choosing uniformly from the
  complete training set and 50% probability of choosing uniformly from the
  weak-rich pool, with replacement.  Distributed rank sharding follows the
  existing infinite-sampler contract.
- The sampler may inspect training annotations only.  Validation labels do not
  influence pool membership or weights.
- Selection uses official CloudSEN validation.  L8 and internal test are unread.

## Pre-registered gates and stop-loss

- validation mIoU >=68.0;
- mean thin-cloud/cloud-shadow IoU >=51.0;
- native-PyTorch FP16 speedup versus frozen V8 >=2.0x;
- deployment parameters <=3.0M;
- all metrics finite and internal test sealed.

If any gate fails, close prevalence-balanced sampling without changing the
quartile, mixture probability, replacement rule, duration, seed, or thresholds.
The next method must address contextual confusion rather than another sampling,
loss, detail, LiteFPN, MobileNet-family, or distillation variant.  If all gates
pass, perform one frozen L8 zero-shot evaluation next.

## Expected outcome

Success requires at least 0.65 validation point over Phase 37 while retaining
its weak-class and deployment gates.  Improved thin-cloud IoU alone is not a
pass.
