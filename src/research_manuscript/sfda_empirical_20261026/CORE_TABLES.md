# Core tables — manuscript version

All results are existing endpoints. Percentages are rounded for presentation; counts are exact. No new experiment or metric recomputation is represented. The internal phase identifiers in provenance notes should be removed from the submission copy after references to the released artifact structure are finalized.

## Table 1. Aggregate gains and class-specific costs of pseudo-label adaptation

| Model | Group mIoU (%) | Pooled mIoU (%) | Surface IoU (%) | Cloud IoU (%) | Shadow IoU (%) | Macro Shadow AP (%) | Pooled Shadow AP (%) | Macro Shadow recall (%) | Macro Shadow FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | 61.280 | 71.689 | 96.197 | 91.121 | 27.749 | 80.206 | 29.809 | 77.437 | 3.957 |
| Q90-PL | 63.418 | 77.743 | 92.916 | 92.994 | 47.317 | 79.082 | 68.577 | 62.298 | 1.066 |
| Difference (pp) | +2.138 | +6.054 | −3.280 | +1.873 | +19.568 | −1.123 | +38.769 | −15.139 | −2.891 |

**Caption.** Source-free LoRA adaptation improves aggregate mIoU in the existing development experiment, while class IoU, ranking metrics and default recall exhibit different changes. Class IoUs are computed from pooled confusion counts. Group metrics use equal group weighting, whereas pooled metrics aggregate pixels; these estimands need not agree.

**Protocol note.** One training seed; 32 development groups; Shadow AP is defined in 7 groups. Recall/FPR use their original positive/negative-support denominators. Shadow scores remain softmax(s)[Shadow]. Classes with zero union are NA; false predictions into an absent class create nonzero union and remain penalized. These are historical development results, not independent confirmation.

**Provenance.** [Phase311 METRICS.csv](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/METRICS.csv), consolidating the verified Source/PL/SP/CM endpoints; [Phase304 report](D:/Cloud-Adapter-light/src/research_results/phase304_20261024/RESULT.md).

### Table 1b. PL replication under a fixed schedule and independent LoRA initializations

| Initialization | Group mIoU (%) | Pooled mIoU (%) | Group mIoU gain (pp) | Macro Shadow recall (%) | Recall change (pp) | Cloud→Shadow net reduction | Surface→Cloud net increase |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30201 | 63.418 | 77.743 | +2.138 | 62.298 | −15.139 | 879,124 | 498,719 |
| 31501 | 63.140 | 76.093 | +1.860 | 50.672 | −26.765 | 839,287 | 80,372 |
| 31502 | 62.050 | 78.025 | +0.770 | 55.841 | −21.597 | 889,726 | 329,219 |
| Mean ± sample SD | 62.869 ± 0.723 | 77.287 ± 1.044 | +1.590 ± 0.723 | 56.270 ± 5.825 | −21.167 ± 5.825 | — | — |

**Caption.** All three initializations improve group mIoU, reduce Cloud→Shadow errors and increase Surface→Cloud errors, while macro Shadow recall decreases. The error magnitudes differ across initializations; direction agreement is not a significance test. Each run has 955 updates and uses the same Source, accepted labels, crops and flips. LoRA A initializations differ and initial B matrices are zero. The experiment tests initialization sensitivity under a fixed schedule, not dataset sampling uncertainty or independent-domain generalization. Deterministic settings retain the recorded CuBLAS warning; complete bitwise determinism is not claimed. Sample SD uses ddof=1 and does not describe uncertainty over independent target domains. SP/CM are not replicated by this table.

**Provenance.** [Phase315 report](D:/Cloud-Adapter-light/src/research_results/phase315_20261026/RESULT.md), [SEED_METRICS.csv](D:/Cloud-Adapter-light/src/research_results/phase315_20261026/SEED_METRICS.csv). Independent confirmation pools admitted at this stage: zero.

## Table 2. Score interpretation changes accepted supervision and actual updates

| Acceptance rule at 0.90 | Accepted pixels | Coverage (%) | Accepted Surface / Cloud / Shadow | Products with updates | Updates / attempts |
|---|---:|---:|---|---:|---|
| max softmax(s) | 101,857 | 0.097519 | 101,857 / 0 / 0 | 2 | 6 / 1,000 |
| max q, q=s/Σs | 75,469,158 | 72.25497 | 40,622,307 / 33,554,491 / 1,292,360 | 97 | 955 / 1,000 |

**Caption.** The same numeric threshold does not give comparable supervision under two score interpretations. The denominator is 104,448,397 image-valid pixels from 100 fit products. Accepted pixels are unique full-image pixels; update counts follow the same crop schedule and are not cumulative pixel counts. The first rule has accepted pixels in 5 products but updates in only 2 under that schedule.

**Scope note.** Mask2Former semantic scores s are nonnegative query-aggregated scores, not native class logits. q is a class-score proportion, not a calibrated correctness probability. The comparison diagnoses support and update-budget differences; it is not an equal-update performance ablation or a new confidence-calibration method. The downstream CE(s,pseudo) loss was held unchanged.

**Provenance.** [Phase303 report](D:/Cloud-Adapter-light/src/research_results/phase303_20261024/RESULT.md), [Phase302 report](D:/Cloud-Adapter-light/src/research_results/phase302_20261020/RESULT.md), [Phase304 report](D:/Cloud-Adapter-light/src/research_results/phase304_20261024/RESULT.md).

## Table 3. Directed error changes underlying the aggregate gain

| Reference → predicted class | Source errors | PL errors | Removed old errors | Added errors | Net error change (PL−Source) | Removed errors corrected / changed to another wrong class |
|---|---:|---:|---:|---:|---:|---|
| Surface → Cloud | 134,027 | 632,746 | 27,632 | 526,351 | +498,719 | 27,282 / 350 |
| Surface → Shadow | 108,750 | 47,007 | 65,032 | 3,289 | −61,743 | 45,479 / 19,553 |
| Cloud → Surface | 246,453 | 300,934 | 94,952 | 149,433 | +54,481 | 94,697 / 255 |
| Cloud → Shadow | 1,161,820 | 282,696 | 883,997 | 4,873 | −879,124 | 784,478 / 99,519 |
| Shadow → Surface | 111,058 | 143,465 | 19,153 | 51,560 | +32,407 | 5,660 / 13,493 |
| Shadow → Cloud | 10,311 | 46,107 | 1,789 | 37,585 | +35,796 | 260 / 1,529 |

**Caption.** Errors are decomposed using the fixed tensor T[reference, Source prediction, PL prediction]. Removing a particular error need not correct the pixel: it can create another error. In particular, the net reduction of 879,124 Cloud→Shadow errors differs from the 784,478 pixels actually corrected to Cloud.

**Support note.** All counts use the original common reference support across all 32 groups. They are descriptive pixel counts, not independent samples or significance tests. Newly added Surface→Cloud errors include 506,798 originally correct Surface pixels and 19,553 originally misclassified Shadow predictions on Surface reference pixels. Thus 526,351 new errors, 506,798 new damage events and 498,719 net errors are different quantities.

**Provenance.** [Phase304 report](D:/Cloud-Adapter-light/src/research_results/phase304_20261024/RESULT.md) and its ERROR_EVENTS/TRANSITIONS27 artifacts; [Phase305 decomposition](D:/Cloud-Adapter-light/src/research_results/phase305_20261025/RESULT.md).

## Table 4. Repairing surface damage can revoke cloud corrections

| Adaptation | Group / pooled mIoU (%) | Original D repaired to Surface | D remaining Cloud / moved to Shadow | Original B retained as Cloud | B returned to Shadow / moved to Surface | New Surface→Cloud / Surface→Shadow outside D | Largest-damage group Surface IoU / mIoU (%) |
|---|---|---|---|---|---|---|---|
| Q90-PL | 63.418 / 77.743 | 0 (0.000%) | 506,798 / 0 | 784,478 (100.000%) | 0 / 0 | 0 / 2,939 | 80.270 / 40.135 |
| Q90-SP | 57.079 / 72.748 | 371,012 (73.207%) | 131,290 / 4,496 | 362,132 (46.162%) | 417,682 / 4,664 | 43,277 / 63,493 | 95.168 / 31.723 |
| Q90-CM | 61.565 / 72.514 | 475,414 (93.807%) | 30,425 / 959 | 299,949 (38.235%) | 383,067 / 101,462 | 14,690 / 10,127 | 100.000 / 100.000 |

**Caption.** D consists of 506,798 reference-Surface pixels correctly classified by Source and changed to Cloud by PL. B consists of 784,478 reference-Cloud pixels changed from Source-Shadow to PL-Cloud. Both sets are fixed retrospective evaluation sets and are excluded from training inputs. All three models start independently from the same initial LoRA state. The intervention repairs neither define all errors nor replace the full confusion analysis.

**Intervention note.** SP adds KL matching to the same-view Source distribution in low-accepted-confidence Source-Surface regions. CM replaces this with a one-sided squared penalty on the student Cloud−Surface score margin exceeding the Source margin. The protection mask is identical; all methods have 955 updates. Both penalties use λ=1 but have different scales, so this does not isolate penalty direction at matched strength. SP/CM also require additional teacher forwards; identical update counts do not imply identical computation.

**Interpretation note.** CM restores the historically largest-damage group to the Source result but remains below PL in both aggregate mIoU measures. The chosen group is a post-hoc diagnostic, not an independent endpoint. Its reference is entirely Surface; the original union-based rule is retained. The generalization or inevitability of these tradeoffs has not been established.

**Provenance.** [Phase311 METRICS.csv](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/METRICS.csv), [DAMAGE_BENEFIT.csv](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/DAMAGE_BENEFIT.csv), and [Phase311 report](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/RESULT.md).

## Material assigned to the supplement

- All 32 per-group results, denominators, union-class counts and missing-class cases; complete 27-cell tensors and boundary/excluded support.
- Same-512-view probes, frozen-parameter/gradient/initialization checks and actual forward costs.
- Omni preliminary experiments, explicitly separated from the VFM comparison.
- Supervised multispectral probe versus pseudo-label-only selector, preserving the distinction between information diagnosis and SFDA gain.
- Geometry readiness and dual-annotation limitations, without claiming a positive geometry or annotation-shift result.

Supplement placement does not hide failed methods: SP/CM remain in the main table. Remaining branches explain why they do not support the current central claim and need not become separate claimed contributions.
