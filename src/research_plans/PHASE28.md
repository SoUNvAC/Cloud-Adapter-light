# Phase 28 — Landsat-8 Biome zero-shot cross-sensor evaluation

## Motivation and scope

Phase 27 established a stable compact operating point on CloudSEN12 L1C. Before
the internal CloudSEN test is unsealed, Phase 28 tests whether its efficiency
gain survives a sensor and geographic distribution shift. No model is trained
or selected on Landsat-8 Biome. All three clean V8 checkpoints and all three
frozen ResNet-18 checkpoints are evaluated zero-shot on the complete L8 Biome
test split.

This dataset was present in the historical Phase 1-20 workspace, so it is an
external cross-sensor benchmark rather than a never-observed dataset. The
current architectures and thresholds were frozen before their L8 results were
observed. This limitation must be disclosed and the experiment must not be
described as a pristine external holdout.

## Fixed taxonomy mapping

CloudSEN output IDs are `(clear, thick cloud, thin cloud, cloud shadow)`. L8
Biome IDs are `(clear, cloud shadow, thin cloud, cloud)`. Before any metric is
computed, predicted IDs are mapped by `[0, 3, 2, 1]`; thus thick cloud becomes
the L8 cloud class and shadow/thin retain their semantic identities. No
confidence threshold, calibration, test-time augmentation, or per-model mapping
is allowed.

## Pre-registered gates and stop-loss

- Every ResNet-18 seed must reach at least 35.0 external mIoU.
- Mean ResNet-18 external mIoU may trail mean V8 by at most 6.5 points, matching
  the accepted internal efficiency trade-off scale.
- ResNet-18 external mIoU sample standard deviation must be <=1.0.
- All six evaluations must complete with finite metrics.
- Internal CloudSEN test remains sealed.

If these gates pass, Phase 29 may unseal the internal CloudSEN test exactly once
for the frozen V8 and ResNet-18 three-seed models. If they fail, Phase 29 closes
ResNet-18 and evaluates a separately pretrained mobile-backbone direction; no
L8-driven repair is allowed.

## One-click command

```bash
cd src
bash tools/run_phase28_l8_external_oneclick_4090d.sh
```

The report is written to
`work_dirs/phase28_l8_external/PHASE28_REPORT.txt`. The script never starts
Phase 29.
