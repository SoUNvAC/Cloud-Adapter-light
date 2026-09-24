# Phase 55 — Thin/Shadow asymmetric negative-transfer audit

## Motivation

On the corrected USGS `Shadows?=yes` target-val subset, Phase 52A-fix changed
thin-cloud IoU from 11.7114 to 37.4599 (+25.7485) while changing shadow IoU
from 26.7183 to 1.0008 (-25.7175). The almost exactly conserved weak-class
mean is a specific asymmetric-transfer observation, not evidence that both
weak classes are generally difficult. Phase 55 is a read-only mechanism audit;
it does not train or authorize a replacement segmentation module.

## Frozen data and checkpoints

- source-only: Phase 22 seed-42 iteration 40,000;
- adapted: Phase 52A-fix seed-52 selected iteration 1,000;
- diagnostic training pool: the unchanged Phase 50 selection, restricted to
  the 12 patches whose official USGS scene field is `Shadows?=yes`;
- held-out probe evaluation: all 963 patches from the eight scene-disjoint
  target-val scenes whose official field is `Shadows?=yes`;
- target-test and CloudSEN internal test remain sealed.

No result may change the selection, checkpoint, labels, feature layer, ridge
strength, subspace rank rule or thresholds below.

## Audit A — per-injection thin/shadow gradient geometry

At the final Phase 52A-fix checkpoint, form separate mean negative-log semantic
probability losses for source-order thin class 2 and shadow class 3. Only
training patches containing both classes at the score resolution contribute.
At injection blocks `[2, 5, 8, 11]`, measure cosine similarity between the two
loss gradients with respect to (a) the post-MsRE token tensor and (b) the
layer-specific learnable-token slice. Report pooled cosine, per-image values
and a deterministic scene/patch bootstrap 95% interval.

The gradient-conflict gate passes only if the pooled post-MsRE cosine is below
`-0.10` at at least two of four blocks and the mean pooled cosine across all
four blocks is below zero. This gate is descriptive and cannot be rescued by
changing loss normalization or choosing a subset after inspection.

## Audit B — held-out layerwise shadow linear probes

For both source-only and Phase 52A-fix, collect the 384-dimensional patch
tokens after the source Cloud-Adapter and, for the candidate, after any MsRE
injection at each of all 12 DINOv2-S blocks. Target head-delta is excluded so
the probe isolates backbone representation. Labels are nearest-neighbour
reduced to the patch grid. A class-balanced ridge linear probe with fixed
`lambda=0.01` is fitted independently at every layer on the 12 valid selected
training patches and evaluated without refitting on the 963 corrected
target-val patches. Primary separability is pooled shadow-vs-non-shadow AUROC;
balanced accuracy and zero-threshold IoU are secondary. Per-scene AUROC and a
seed-55 scene bootstrap interval for candidate-minus-source are reported.

The representation-loss gate passes only if candidate shadow AUROC is at
least `0.05` below source-only at an injection layer and does not recover to
within `0.02` of source-only at layer 11. This distinguishes information loss
from a decoder-only decision-boundary failure.

## Audit C — thin-update versus shadow Fisher/Jacobian subspaces

For every valid diagnostic training patch, flatten the gradients of the same
class-specific losses with respect to all 380,577 target parameters and
normalize each per-image row. The thin gradient row span defines the observed
thin update subspace; the shadow gradient row span defines an empirical
Fisher/Jacobian subspace. Each basis retains 90% singular-value energy, capped
at rank 8. Report all canonical cosines/principal angles, the smallest angle,
the fraction of shadow gradient energy projected into the thin subspace, and
the cosine between aggregate thin and shadow gradients.

The overlap gate passes only if the smallest principal angle is at most 75
degrees, at least 10% of shadow gradient energy lies in the retained thin
subspace, and the aggregate signed gradient cosine is negative.

## Decision rule

Only if all three gates pass may the project state that the Phase 52A-fix
trade-off is supported by a thin/shadow representation-level gradient-conflict
mechanism and preregister a decoupled module. Any failed gate executes
`stop_thin_shadow_conflict_hypothesis`: the result remains a documented
asymmetric negative-transfer phenomenon, but no conflict-specific architecture
may be designed from these data. No threshold, rank, regularizer or probe
variant may be swept after results are visible.

## Result and stop decision

The complete audit used the existing `cloud-lite-pt210` environment on the
RTX 4090 D; the requested name `cloud-lite-210` does not exist on the host.
Five of the 12 official `Shadows?=yes` selected patches contained both thin
and shadow at the diagnostic score resolution and contributed gradients. All
963 corrected target-val patches from eight scenes contributed to the probes.
The result artifact SHA-256 is
`ea3c3566487472d266d0f060eed1b1b0e53d3321af0551e84a314bdde7f71c83`.

The pooled post-MsRE thin/shadow gradient cosines at blocks 2, 5, 8 and 11
were `+0.1628`, `+0.3860`, `+0.2518` and `-0.0018`, with a four-layer mean of
`+0.1997`. Only the per-patch layer-11 mean was clearly negative (`-0.1233`,
bootstrap 95% interval `[-0.1659, -0.0711]`); it did not survive equal-patch
gradient pooling, and no two blocks crossed the preregistered `-0.10` line.
The gradient-conflict gate failed.

The held-out shadow-probe AUROC candidate-minus-source differences at the four
injection blocks were `-0.0006`, `-0.0295`, `+0.0254` and `+0.0425`. The final
candidate representation was therefore more, not less, linearly separable for
shadow than source-only (`0.6383` versus `0.5958`). Neither the required
`-0.05` injection-layer loss nor persistent layer-11 loss occurred, so the
representation-information-loss gate failed.

The retained thin and shadow parameter-gradient subspaces both had rank 3.
Their principal angles were `20.82`, `64.31` and `86.62` degrees, and 43.91%
of normalized shadow-gradient energy projected into the thin subspace. This
establishes substantial overlap, but its signed aggregate gradient cosine was
`+0.4337`, not negative. The subspace-conflict gate therefore failed.

All three mechanism gates failed and Phase 55 executes
`stop_thin_shadow_conflict_hypothesis`. The Phase 52A-fix IoU exchange remains
a real asymmetric negative-transfer result, but these measurements reject the
specific claim that MsRE erased shadow information or that thin and shadow
updates are globally antagonistic. The evidence instead localizes the collapse
after a still shadow-informative backbone representation, plausibly in the
shared head/readout competition; that is an interpretation, not authorization
for another module. Target-test and CloudSEN internal test remained sealed.
