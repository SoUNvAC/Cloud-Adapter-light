# Phase 54 — independent shadow information and label audit

Phase 54 is diagnostic and independent of Phase 52 training. It does not use a
Phase 52 checkpoint as an initialization and it is not reported as a domain
adaptation result. The frozen RGB model is the Phase 22 seed-42 checkpoint. The
only labelled target training data are the 65 patches selected without labels
in Phase 50. The scene-disjoint `target_val` split is used for model selection
and reporting. `target_test` and the CloudSEN internal test remain sealed.

## A. Information increment oracle

Five probes use the same architecture, parameter count, optimizer, sampler,
augmentations, number of updates, seeds, train patches, validation images and
checkpoint rule. The probe input always reserves all auxiliary slots; a
condition that does not expose a slot fills it with zero. Therefore adding a
modality never adds a trainable parameter.

The common input is the frozen Phase 22 output-index-2 RGB VFM feature. The
incremental conditions are:

1. RGB VFM feature;
2. RGB feature + Landsat B5 NIR;
3. RGB feature + B5 NIR + B6/B7 SWIR;
4. the above + MTL sun elevation and azimuth encodings;
5. the above + a non-local neighbouring-cloud probability computed from the
   frozen Phase 22 thin-plus-thick probability, without using labels.

The frozen TorchGeo archive stores all 11 bands as a derived uint8 product, not
the original Landsat DN product; applying the original MTL reflectance
coefficients again would therefore be invalid. Optical channels use the
archive's fixed `value / 255` encoding, while MTL is used only for sun elevation
and azimuth. Every raw window must reconstruct its existing RGB PNG to mean
absolute error at most 1/255 before it is eligible. The cloud-context channel is
a fixed annular average and excludes the centre neighbourhood. The probe changes only shadow versus non-shadow; the
frozen model's conditional clear/thin/thick ratios are retained. It is trained
with the same balanced shadow BCE, Tversky and one-pixel boundary losses in all
conditions. Seeds are 42, 123 and 3407. Training length is fixed at 10,000
updates, validation is every 1,000 updates, and the learning-rate schedule ends
at update 10,000. Each seed selects the checkpoint with highest validation
shadow IoU; ties are broken by thin-cloud IoU and then mIoU. This longer
diagnostic schedule is intentionally separate
from the Phase 52 4,000-update stop-loss protocol.

The audit passes only if all of the following hold:

- exactly 65 labelled train patches and 1,905 validation patches are used in
  every condition;
- all five probes have exactly the same trainable parameter count;
- relative to RGB-only, at least one spectral/geometric condition improves
  mean shadow IoU by at least 5.0 points across the three seeds;
- that condition improves shadow IoU by at least 2.0 points in every seed;
- its mean thin-cloud IoU is no more than 1.0 point below RGB-only;
- all metrics are finite and both sealed tests remain unread.

Failure stops the information-increment route. No condition, seed, checkpoint,
normalization, receptive field or threshold may be changed after results are
observed.

## B. Label consistency audit

Before viewing any proposed corrections, deterministically select 144
`target_val` patches. Selection prioritizes the absolute shadow-probability and
shadow-mask disagreement between the frozen Phase 22 model and Phase 52A, then
enforces coverage of all eight biomes, three sun-elevation strata and explicit
quotas for water, snow/ice (terrain proxy) and urban scenes. Selection uses no
revised labels.

Each patch is reviewed against the same written rule with co-registered RGB,
NIR, SWIR, predicted masks, original label and solar geometry. A valid revised
mask must have a reviewer identifier, completion timestamp and reason codes for
every edited shadow component. Reviewers are blinded to aggregate IoU changes.
Ambiguous components remain unchanged and are marked ambiguous; model output is
never copied into the reference mask. A second reviewer adjudicates every edit
and a disagreement record is retained.

The audit reports original-label and adjudicated-label shadow IoU for both
frozen models, missed-shadow pixels/components, false-shadow pixels/components,
and signed boundary displacement. B passes (meaning label inconsistency is
material) if either:

- adjudication changes either model's shadow IoU by at least 5.0 points; or
- at least 10% of original shadow pixels are removed as false shadow, at least
  10% of adjudicated shadow pixels were originally missed, or the median
  absolute matched-boundary displacement is at least 2 pixels.

Until the 144 masks have complete independent review and adjudication, B is
`incomplete`, never pass or fail. Automated spectral thresholds, model
predictions and QA products are not accepted as substitute ground truth.
