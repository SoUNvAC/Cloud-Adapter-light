# Cloud–Shadow Confusion and Surface Damage in Source-Free Adaptation of Vision Foundation Models

Assembled internal empirical manuscript draft, updated through Phase316 on 2026-10-26. PL has three independent LoRA initialization runs under one fixed schedule; SP/CM remain single-initialization interventions. No independent-pool result is implied. This is a complete reading draft of the available study, not a submission-ready paper. Tables are maintained in [CORE_TABLES.md](D:/Cloud-Adapter-light/src/research_manuscript/sfda_empirical_20261026/CORE_TABLES.md) and reproduced inline in this reading version.

## Abstract

Source-free adaptation offers a way to transfer cloud segmentation models using unlabeled target imagery, but aggregate gains can conceal different changes in cloud, cloud-shadow and surface errors. We study a controlled vision-foundation-model configuration using 100 fit products and 32 development groups. Across three independent LoRA initializations with a fixed schedule, pseudo-label adaptation improves group-mean mIoU by 0.770–2.138 percentage points. Cloud→Shadow errors decrease and Surface→Cloud errors increase in all runs, while macro Shadow recall drops by 15.139–26.765 points. In the original run, two Source-preserving interventions repair 73.207% and 93.807% of newly damaged Surface pixels but retain only 46.162% and 38.235% of the original Cloud corrections. Neither improves the overall mIoU tradeoff over that pseudo-label baseline, although one completely restores the largest-damage group. We also document how score interpretation changes accepted supervision and actual update counts. These development-set observations motivate reporting damage repair alongside benefit retention and class-specific metrics. Initialization repetition supports the baseline observation under the fixed protocol; intervention repetition and independent-domain confirmation remain absent. The results do not establish an inevitable tradeoff or a new adaptation algorithm.

## 1. Introduction

Cloud segmentation commonly distinguishes visible cloud, its shadow and unobscured surface. These categories also define different failure modes. Predicting a cloud as Shadow is not the same error as predicting clear Surface as Cloud. A model update may reduce one error while increasing the other, so a higher aggregate segmentation score does not specify which observations have become more reliable.

Source-free domain adaptation provides a setting in which a pretrained model is adapted using unlabeled target imagery without revisiting source training samples. Such adaptation is relevant when only a deployed checkpoint is available. However, removing target labels from the training objective does not remove the need to examine the classes and regions receiving pseudo-label supervision. Nor does it make a repeatedly used development set an independent test set.

Vision foundation models and lightweight adaptation make this question practical for cloud segmentation. Cloud-Adapter already studies a frozen VFM with a lightweight cloud-segmentation adapter [1]. Here we reuse that model family and investigate target adaptation rather than introduce another backbone or adapter. Our focus is the relationship between aggregate improvement, directed class-error changes and interventions intended to protect previously correct predictions.

The distinction matters when evaluating a repair. If an intervention restores many Surface pixels but revokes Cloud corrections obtained by adaptation, its repair count alone overstates the outcome. Conversely, a failure to improve overall mIoU does not mean that the intervention repaired nothing. A useful analysis must retain both observations and account for new errors outside the originally damaged region.

We study one controlled three-class VFM configuration using fixed pseudo-label adaptation. The baseline is repeated with three independent LoRA initializations under the same data and crop schedule. Two preservation interventions are examined at the original initialization. We ask how aggregate gains relate to directed errors, whether repairing Surface damage retains the baseline's Cloud corrections, and how score interpretation and evaluation conventions affect those comparisons.

The study contributes a documented empirical case and an associated reporting procedure. First, it decomposes aggregate adaptation results into corrections, damage and error replacement. Second, it evaluates Surface repair jointly with retention of a fixed set of Cloud corrections, rather than treating either set as the entire task. Third, it reports accepted supervision, actual optimization updates, class support and initialization variation. The observations are restricted to the tested configuration and historical development pool. They do not establish a new adaptation algorithm or an unavoidable conflict between class objectives.

## 2. Related work

### 2.1 Cloud segmentation and domain adaptation

Cloud-Adapter [1] uses a frozen vision foundation model and lightweight trainable components for cloud segmentation. Its role here is an existing architecture on which to study target adaptation. We do not claim frozen VFM adaptation, cloud segmentation with an adapter, or low-parameter tuning as new ideas.

Domain adaptation for cloud detection also predates this study. Du et al. [2] examine multispectral satellite-borne cloud detection with offline adaptation and online test-time adaptation, including CNN-based processing. Their mission-oriented setting and cloudiness classification differ from our pixel-level Surface/Cloud/Shadow analysis. The settings therefore motivate the problem but do not provide directly comparable numbers or establish that adaptation of a VFM is itself novel.

### 2.2 Source-free learning and pseudo-label supervision

SHOT [3] studies transferring a trained source hypothesis without source data, combining target representation learning with information maximization and pseudo-labeling. In remote-sensing segmentation, Diffusion-Guided Label Enrichment [4] investigates improving pseudo-label coverage and quality from incomplete high-quality seeds. These works establish a broader context for source-free learning and pseudo-label design.

Our experiment uses a fixed teacher and fixed accepted labels, with no iterative label enrichment, diffusion model or claimed reproduction of SHOT. The accepted supervision and actual update counts are reported explicitly. Because competing methods are not implemented under the same protocol, we make no state-of-the-art claim or numerical comparison with their published benchmarks. A broader baseline study would be required for such a comparison.

### 2.3 Representation, segmentation and low-rank tuning

DINOv2 [5] provides self-supervised visual representations; Mask2Former [6] supplies a query-based formulation for image segmentation. LoRA [7] adapts pretrained linear transformations through trainable low-rank updates. These are reused components of the controlled configuration, not contributions of the present study. Their combination also makes implementation details relevant: a query-aggregated semantic score should not be silently treated as a calibrated per-pixel correctness probability.

### 2.4 Annotation and evaluation scope

The Sentinel-2 Cloud Mask Catalogue [8] provides three-class masks and documents annotation uncertainty and multiple-annotator subsets. These properties motivate retaining the published class definitions and reporting missing-class support. They do not justify replacing difficult reference pixels or treating annotator agreement as physical truth. Our dual-reference audit is used to delimit what can be tested; it does not support an annotation-shift method.

The cited literature positions the study but is not an exhaustive survey. The intended distinction is an empirical analysis of class-transfer and repair costs in one VFM adaptation setting. It should not be read as a claim that directed error analysis, benefit retention or annotation-sensitive evaluation has never been considered elsewhere.

## 3. Experimental protocol

### 3.1 Model and adaptation

The controlled Source model uses DINOv2-S with the existing Cloud-Adapter and a three-parent Mask2Former head. The Source checkpoint was selected using source validation data. The model predicts Surface, Cloud and Shadow. The Source checkpoint, existing adapter, neck, head and buffers remain frozen during target adaptation. A fresh rank-4 LoRA with 294,912 trainable parameters is attached to the attention and MLP linear layers in all 12 backbone blocks.

The target imagery comes from the Sentinel-2 Cloud Mask Catalogue [8], whose released subscenes have 1,022×1,022 pixels at a nominal 20 m resolution. We retain the existing representative-product and group assignments rather than constructing a new favorable subset. The Source was trained and selected using the recorded CloudSEN12 source splits [9]. Stored source/target audit records do not establish an exhaustive non-overlap guarantee for the foundation model's pretraining data.

The target fit set contains 100 representative products; evaluation uses the existing 32 development groups. The original PL/SP/CM comparison uses initialization 30201 and a matched crop schedule, horizontal flips, optimizer and final-attempt endpoint. PL is additionally repeated with fresh LoRA initializations 31501 and 31502, preserving that schedule and all hyperparameters. LoRA A matrices differ, whereas initial B matrices remain zero; the repetition does not resample products or augmentations. AdamW uses a learning rate of 10⁻⁴ and weight decay of 0.01. Of 1,000 scheduled attempts, 955 contain accepted pseudo-labels and trigger an update. Empty attempts remain skipped rather than being replaced. Training uses image-derived validity and does not read target reference masks.

Full-image teacher and evaluation scores are assembled from nine 512-pixel windows with row and column starts {0,256,510}, averaging semantic scores in overlaps. We preserve the original RGB input scaling and preprocessing. The exact crop schedule, scores and frozen-parameter checks belong in the reproducibility supplement.

RGB uses B04/B03/B02 from the original TOA cube, multiplied by 255 without clipping reflectance above one. Validity is determined from image values, including finite-value, zero and fill checks; it does not use reference-mask validity in the training loader. The existing BGR-to-RGB preprocessing chain is retained exactly. These choices define the tested model interface and are not proposed as a universal Sentinel-2 normalization rule.

### 3.2 Semantic scores and accepted supervision

For class c, the Mask2Former semantic score is s_c=Σ_j p_j(c)u_j, where p_j is the query class softmax, u_j is the query mask sigmoid, and the no-object class is excluded. These nonnegative aggregate scores are not native per-pixel logits.

Let A be the accepted-pixel set and yhat the frozen teacher argmax. The pseudo-label loss is L_PL=|A|⁻¹Σ_(i∈A) CE(s_student(i),yhat(i)). The same loss implementation is used in all formal adaptation arms. It is a deliberate part of the tested protocol; changing acceptance does not change this objective to negative log q.

Q90-PL accepts the argmax class when max_c q_c≥0.90, where q_c=s_c/Σ_k s_k and the score sum is positive. The student minimizes CE(s,pseudo) on accepted pixels, following the existing implementation. q defines acceptance only and is not a calibrated correctness probability. The Shadow ranking score remains softmax(s)[Shadow], distinct from the acceptance statistic.

The earlier use of max softmax(s) at 0.90 is reported as a support audit, not as an equal-update performance baseline. Changing acceptance also changed the available classes and the actual update count.

### 3.3 Source-preserving interventions

Both interventions use the same image-only mask P: Source predicts Surface, its full-image qSurface is below 0.90, and the input is valid. P is disjoint from Q90-accepted pixels. The teacher target is computed on the same 512-pixel crop and flip as the student, avoiding a full-context versus crop mismatch in the preservation target.

SP adds the mean KL divergence from the frozen Source softmax(s) distribution to the student distribution on P. CM instead adds the mean squared positive excess of the student Cloud−Surface score margin over the Source margin. Each uses a fixed coefficient of one, and each is averaged over all P pixels. The pseudo-label loss and update schedule are unchanged. These penalties have different scales; equal coefficients do not constitute strength matching. CM has no direct dependence on the Shadow output score, but shared parameters and class competition can still change Shadow predictions.

More explicitly, SP uses L_SP=mean_P KL(softmax(s_teacher) || softmax(s_student)). CM uses L_CM=mean_P [max(0,m_student−m_teacher)]², where m=s_Cloud−s_Surface. Each arm minimizes L_PL plus its respective penalty. Empty P yields zero penalty, while empty A skips the complete update. Consequently, no preservation-only updates are added to the fixed 955-step schedule.

### 3.4 Evaluation and retrospective event sets

All original groups and missing-class conventions are retained. Group-mean and pooled mIoU are reported together with pooled class IoUs and Shadow ranking and operating-point metrics. A class with zero union has undefined IoU; false predictions into an absent class create a nonzero union and remain penalized. Shadow AP is defined in only seven of the 32 groups. Recall and FPR retain their respective support conditions.

We decompose predictions with T[y,p_Source,p_PL]. A removed directed error is distinguished from a correction to the reference class. Two fixed retrospective sets summarize the interventions: D contains reference-Surface pixels correctly predicted by Source and predicted as Cloud by PL; B contains reference-Cloud pixels predicted as Shadow by Source and correctly as Cloud by PL. D and B are used only after endpoint sealing. They do not define training masks, hyperparameters or sample selection. Errors introduced outside D and corrections outside B are reported separately.

For an intervention, damage-repair rate is the fraction of D predicted as Surface; benefit-retention rate is the fraction of B still predicted as Cloud. Remaining Cloud errors and transitions to Shadow within D are reported separately, as are returns to Shadow and moves to Surface within B. These conditional rates do not replace full-image evaluation. Their denominators are fixed by the original PL endpoint and must not be redefined for each intervention.

### 3.5 Evidence, aggregation and reproducibility

The baseline seeds are 30201, 31501 and 31502. We report each endpoint, the arithmetic mean and sample standard deviation with ddof=1. The crops, flips, teacher labels and Source checkpoint are shared, so the runs test initialization sensitivity rather than independent target-domain variation. The Source has one fixed value, not a fabricated seed standard deviation. We do not treat pixels or the 32 groups repeated across three seeds as independent extra samples.

Source weights and frozen buffers are checked before and after adaptation. Initial zero-LoRA predictions must match Source scores, and replayed historical endpoints must match their cached predictions before comparison. Error tensors are checked against confusion-matrix marginals. Actual calls include training, probes, endpoint evaluation, validation gates and any recorded recovery. Deterministic settings retain the recorded CuBLAS warning; bitwise determinism across all operations is not asserted.

Target references are read for evaluation after endpoint sealing, but the development pool has been repeatedly inspected during the broader study. We therefore describe the results as exploratory development evidence, even though the adaptation objective itself uses no target labels. The original intervention comparison remains single-initialization evidence.

## 4. Results

### 4.1 Aggregate improvement accompanies different class-level changes

Q90-PL raises group-mean mIoU from 61.280% to 63.418% and pooled mIoU from 71.689% to 77.743% (Table 1). Cloud and Shadow pooled IoUs increase, whereas Surface IoU falls from 96.197% to 92.916%. The Shadow metrics do not indicate a uniform improvement: pooled AP rises from 29.809% to 68.577%, while macro AP decreases from 80.206% to 79.082%. At the default decision rule, macro recall falls from 77.437% to 62.298% and macro FPR from 3.957% to 1.066%.

These metrics summarize different supports and weighting schemes. Their divergence is consistent with heterogeneous group-level changes, but does not by itself identify their cause. In particular, the observations do not establish that adaptation only moves an operating point, or that ranking improves uniformly across groups. Group-level values and valid denominators are retained in the experiment artifacts alongside the aggregate results.

The two additional initializations retain positive group-mIoU gains of 1.860 and 0.770 points (Table 1b). Across all three runs, group mIoU is 62.869 ± 0.723% and pooled mIoU is 77.287 ± 1.044% (mean ± sample SD). Cloud→Shadow errors decrease by 839,287–889,726 pixels, whereas Surface→Cloud errors increase by 80,372–498,719. Macro Shadow recall decreases in every run, by 15.139–26.765 points. Thus the observed gain and two directed error changes repeat under these initializations, but their magnitudes are not invariant. The same products, crops and pseudo-labels are reused; this is not confirmation on an independent target pool or evidence over data-sampling uncertainty.

#### Table 1. Aggregate gains and class-specific costs of pseudo-label adaptation

| Model | Group mIoU (%) | Pooled mIoU (%) | Surface IoU (%) | Cloud IoU (%) | Shadow IoU (%) | Macro Shadow AP (%) | Pooled Shadow AP (%) | Macro Shadow recall (%) | Macro Shadow FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | 61.280 | 71.689 | 96.197 | 91.121 | 27.749 | 80.206 | 29.809 | 77.437 | 3.957 |
| Q90-PL | 63.418 | 77.743 | 92.916 | 92.994 | 47.317 | 79.082 | 68.577 | 62.298 | 1.066 |
| Difference (pp) | +2.138 | +6.054 | −3.280 | +1.873 | +19.568 | −1.123 | +38.769 | −15.139 | −2.891 |

**Caption.** Source-free LoRA adaptation improves aggregate mIoU in the existing development experiment, while class IoU, ranking metrics and default recall exhibit different changes. Class IoUs are computed from pooled confusion counts. Group metrics use equal group weighting, whereas pooled metrics aggregate pixels; these estimands need not agree.

**Protocol note.** One training seed; 32 development groups; Shadow AP is defined in 7 groups. Recall/FPR use their original positive/negative-support denominators. Shadow scores remain softmax(s)[Shadow]. Classes with zero union are NA; false predictions into an absent class create nonzero union and remain penalized. These are historical development results, not independent confirmation.

**Provenance.** [Phase311 METRICS.csv](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/METRICS.csv), consolidating the verified Source/PL/SP/CM endpoints; [Phase304 report](D:/Cloud-Adapter-light/src/research_results/phase304_20261024/RESULT.md).

#### Table 1b. PL replication under a fixed schedule and independent LoRA initializations

| Initialization | Group mIoU (%) | Pooled mIoU (%) | Group mIoU gain (pp) | Macro Shadow recall (%) | Recall change (pp) | Cloud→Shadow net reduction | Surface→Cloud net increase |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30201 | 63.418 | 77.743 | +2.138 | 62.298 | −15.139 | 879,124 | 498,719 |
| 31501 | 63.140 | 76.093 | +1.860 | 50.672 | −26.765 | 839,287 | 80,372 |
| 31502 | 62.050 | 78.025 | +0.770 | 55.841 | −21.597 | 889,726 | 329,219 |
| Mean ± sample SD | 62.869 ± 0.723 | 77.287 ± 1.044 | +1.590 ± 0.723 | 56.270 ± 5.825 | −21.167 ± 5.825 | — | — |

**Caption.** All three initializations improve group mIoU, reduce Cloud→Shadow errors and increase Surface→Cloud errors, while macro Shadow recall decreases. The error magnitudes differ across initializations; direction agreement is not a significance test. Each run has 955 updates and uses the same Source, accepted labels, crops and flips. LoRA A initializations differ and initial B matrices are zero. The experiment tests initialization sensitivity under a fixed schedule, not dataset sampling uncertainty or independent-domain generalization. Deterministic settings retain the recorded CuBLAS warning; complete bitwise determinism is not claimed. Sample SD uses ddof=1 and does not describe uncertainty over independent target domains. SP/CM are not replicated by this table.

**Provenance.** [Phase315 report](D:/Cloud-Adapter-light/src/research_results/phase315_20261026/RESULT.md), [SEED_METRICS.csv](D:/Cloud-Adapter-light/src/research_results/phase315_20261026/SEED_METRICS.csv). Independent confirmation pools admitted at this stage: zero.

### 4.2 The acceptance statistic determines usable supervision

Applying a 0.90 threshold to max softmax(s) accepts only 101,857 pixels, or 0.097519% of image-valid support (Table 2). All accepted labels are Surface, and only six scheduled attempts trigger updates. The Q90 score-proportion rule accepts 75,469,158 pixels, covering 72.25497% of image-valid support and all three classes; 955 attempts trigger updates.

The example s=[1,0,0] gives a maximum softmax value of approximately 0.576 despite a single nonzero class score. This illustrates why a high numeric threshold on these aggregate scores can have a different meaning from a threshold on native logits. The support audit explains why actual accepted classes and update counts must be reported. It does not establish calibration or attribute a performance difference solely to the choice of score statistic.

#### Table 2. Score interpretation changes accepted supervision and actual updates

| Acceptance rule at 0.90 | Accepted pixels | Coverage (%) | Accepted Surface / Cloud / Shadow | Products with updates | Updates / attempts |
|---|---:|---:|---|---:|---|
| max softmax(s) | 101,857 | 0.097519 | 101,857 / 0 / 0 | 2 | 6 / 1,000 |
| max q, q=s/Σs | 75,469,158 | 72.25497 | 40,622,307 / 33,554,491 / 1,292,360 | 97 | 955 / 1,000 |

**Caption.** The same numeric threshold does not give comparable supervision under two score interpretations. The denominator is 104,448,397 image-valid pixels from 100 fit products. Accepted pixels are unique full-image pixels; update counts follow the same crop schedule and are not cumulative pixel counts. The first rule has accepted pixels in 5 products but updates in only 2 under that schedule.

**Scope note.** Mask2Former semantic scores s are nonnegative query-aggregated scores, not native class logits. q is a class-score proportion, not a calibrated correctness probability. The comparison diagnoses support and update-budget differences; it is not an equal-update performance ablation or a new confidence-calibration method. The downstream CE(s,pseudo) loss was held unchanged.

**Provenance.** [Phase303 report](D:/Cloud-Adapter-light/src/research_results/phase303_20261024/RESULT.md), [Phase302 report](D:/Cloud-Adapter-light/src/research_results/phase302_20261020/RESULT.md), [Phase304 report](D:/Cloud-Adapter-light/src/research_results/phase304_20261024/RESULT.md).

### 4.3 Directed events distinguish correction from error replacement

The largest decrease in a directed error is Cloud→Shadow, which falls by 879,124 pixels (Table 3). Of 883,997 removed errors of that type, 784,478 are corrected to Cloud and 99,519 move to another wrong class. In parallel, Surface→Cloud errors increase by 498,719 pixels. The 526,351 newly added errors include 506,798 formerly correct Surface predictions and 19,553 Surface pixels previously predicted as Shadow.

Thus the aggregate gain includes both substantial correction of Cloud/Shadow confusion and damage to previously correct Surface predictions. Neither the net error change nor the count of removed errors alone describes the corresponding benefit. This event decomposition motivates evaluating preservation constraints against fixed correction and damage sets, while retaining full confusion matrices and newly introduced errors elsewhere.

#### Table 3. Directed error changes underlying the aggregate gain

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

### 4.4 Surface repair does not establish a better overall adaptation outcome

SP repairs 371,012 of the 506,798 original D pixels, or 73.207%, but retains only 362,132 of the 784,478 original B corrections, or 46.162% (Table 4). Its group-mean and pooled mIoU are lower than those of PL. CM repairs 475,414 D pixels, or 93.807%, while retaining 299,949 B corrections, or 38.235%. It also introduces errors outside D, and remains below PL in both aggregate mIoU measures.

CM completely restores the historically largest-damage group, whose reference is entirely Surface. Yet its pooled mIoU is 5.228 percentage points below PL. This separates local repair from overall adaptation quality and shows that the unfavorable overall comparison cannot be attributed solely to the missing-class penalty in that one group.

The interventions provide two observed cases of repair accompanied by substantial benefit reversal. They do not prove an unavoidable conflict between Surface protection and Cloud correction. The protection mask can contain reference-Cloud pixels, the penalties have different strengths, and shared parameters propagate updates beyond the protected region. Those factors preclude assigning the observed cost to a single causal mechanism.

#### Table 4. Repairing surface damage can revoke cloud corrections

| Adaptation | Group / pooled mIoU (%) | Original D repaired to Surface | D remaining Cloud / moved to Shadow | Original B retained as Cloud | B returned to Shadow / moved to Surface | New Surface→Cloud / Surface→Shadow outside D | Largest-damage group Surface IoU / mIoU (%) |
|---|---|---|---|---|---|---|---|
| Q90-PL | 63.418 / 77.743 | 0 (0.000%) | 506,798 / 0 | 784,478 (100.000%) | 0 / 0 | 0 / 2,939 | 80.270 / 40.135 |
| Q90-SP | 57.079 / 72.748 | 371,012 (73.207%) | 131,290 / 4,496 | 362,132 (46.162%) | 417,682 / 4,664 | 43,277 / 63,493 | 95.168 / 31.723 |
| Q90-CM | 61.565 / 72.514 | 475,414 (93.807%) | 30,425 / 959 | 299,949 (38.235%) | 383,067 / 101,462 | 14,690 / 10,127 | 100.000 / 100.000 |

**Caption.** D consists of 506,798 reference-Surface pixels correctly classified by Source and changed to Cloud by PL. B consists of 784,478 reference-Cloud pixels changed from Source-Shadow to PL-Cloud. Both sets are fixed retrospective evaluation sets and are excluded from training inputs. All three models start independently from the same initial LoRA state. The intervention repairs neither define all errors nor replace the full confusion analysis.

**Intervention note.** SP adds KL matching to the same-view Source distribution in low-accepted-confidence Source-Surface regions. CM replaces this with a one-sided squared penalty on the student Cloud−Surface score margin exceeding the Source margin. The protection mask is identical; all methods have 955 updates. Both penalties use λ=1 but have different scales, so this does not isolate penalty direction at matched strength. SP/CM also require additional teacher forwards; identical update counts do not imply identical computation.

**Interpretation note.** CM restores the historically largest-damage group to the Source result but remains below PL in both aggregate mIoU measures. The chosen group is a post-hoc diagnostic, not an independent endpoint. Its reference is entirely Surface; the original union-based rule is retained. The generalization or inevitability of these tradeoffs has not been established.

**Provenance.** [Phase311 METRICS.csv](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/METRICS.csv), [DAMAGE_BENEFIT.csv](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/DAMAGE_BENEFIT.csv), and [Phase311 report](D:/Cloud-Adapter-light/src/research_results/phase311_20261025/RESULT.md).

## 5. Discussion and limitations

The current results support an evaluation practice: report repaired damage alongside retained corrections, class-specific operating-point and ranking metrics, and errors newly introduced elsewhere. A single mIoU gain or a successful repair of a visually prominent failure case does not capture this behavior. This is an empirical observation and an audit practice, not a new segmentation architecture or physical model.

Our exploration of additional information did not establish a deployable correction. A supervised multispectral probe showed limited information gains under grouped holdout, whereas a pseudo-label-only selector produced only a small fixed-readout gain and did not repair the largest-damage group. These experiments use different supervision and cannot be combined as evidence of an unsupervised spectral mechanism. Geometry-based cloud–shadow constraints were not evaluated because the required image-to-grid, height and contextual evidence was incomplete.

Dual-annotation analysis also has a restricted interpretation. The four overlapping development products have identical labels under both annotators and no positive Shadow support; they cannot test annotation-disagreement sensitivity. Teacher precision differences in twelve fit products do not supply student generalization evidence. These findings constrain the claims rather than supporting a positive annotation-shift method.

The principal limitations are one controlled VFM configuration, fixed data sampling and augmentation schedule, repeated development exposure, limited Shadow-positive evaluation support and absence of an independent confirmation pool. PL now has three initialization runs; the SP/CM interventions still have only one. Recorded CuBLAS warnings also preclude claiming complete bitwise determinism. The Source was selected using source validation, with the existing target-overlap audit retained; this does not establish an exhaustive pretraining non-overlap guarantee. Ambiguous high-reflectance references and incomplete geographic provenance remain documented without changing labels or removing products.

Accordingly, the manuscript cannot currently claim stable cross-domain improvement, universal error coupling or a TGRS-ready new method. The baseline class-transfer pattern has repeated under the fixed initialization experiment. The next missing evidence is independent-pool confirmation; the completed admission audit identified no currently eligible pool. If the study expands the intervention conclusion beyond the original endpoints, the interventions themselves require replication; repeated PL alone cannot establish intervention robustness. No additional training follows automatically from the admission failure.

An additional dataset audit did not resolve that limitation. KappaSet [10] provides cloud and shadow class definitions compatible with a documented parent-class mapping, but the available records did not establish exact original-product correspondence, reliable subscene footprints, stored input scaling and actual positive-Shadow evaluation support for an eligible subset. We did not run the model on that dataset or download the full archive to obtain a favorable external result. Its exclusion from the current study means that external confirmation is absent; it says nothing about how the model would perform on KappaSet.

The practical implication is to separate two decisions. Choosing an adapted model for use requires the application's tolerance for missed shadows and false cloud predictions. Establishing a general scientific claim requires evidence beyond this configuration and development pool. Neither decision can be made from the aggregate mIoU gain alone. The current report documents the observed tradeoffs and the unsuccessful repairs without claiming that an untested model, threshold or physical constraint would resolve them.

## 6. Conclusion

In the existing cloud-segmentation experiment, source-free pseudo-label adaptation improves aggregate mIoU while correcting Cloud/Shadow confusion and damaging previously correct Surface predictions. Two preservation interventions repair much of that damage but revoke many original Cloud corrections and do not improve the overall mIoU comparison. The evidence favors evaluating adaptation through both benefit retention and damage repair, while keeping its configuration-specific and exploratory status explicit.

## Data and artifact statement

The underlying Catalogue and CloudSEN12 datasets are identified in [8] and [9]. The internal experiment records retain product/group lists, input and checkpoint hashes, initial/final LoRA weights, accepted-label masks, schedules, cached predictions, directed-event tables and resource/failure logs. Dataset access and redistribution remain subject to the original licenses. This draft does not assert that all internal artifacts have been publicly released or that their full training coverage can be independently certified from the manuscript alone.

## References

1. X. Zou et al., “Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images,” arXiv:2411.13127, 2024. [Author manuscript](https://arxiv.org/abs/2411.13127).
2. A. Du, A.-D. Doan, Y. W. Law and T.-J. Chin, “Domain Adaptation for Satellite-Borne Multispectral Cloud Detection,” Remote Sensing, vol. 16, no. 18, article 3469, 2024. [Publisher article](https://doi.org/10.3390/rs16183469).
3. J. Liang, D. Hu and J. Feng, “Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation,” ICML, PMLR 119, pp. 6028–6039, 2020. [Proceedings](https://proceedings.mlr.press/v119/liang20a.html).
4. W. Liu, H. Liu, L. Zhang and B. Fan, “Source-Free Domain Adaptive Semantic Segmentation of Remote Sensing Images with Diffusion-Guided Label Enrichment,” arXiv:2509.18502, 2025. [Author manuscript](https://arxiv.org/abs/2509.18502).
5. M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” arXiv:2304.07193, 2023, revised 2024. [Author manuscript](https://arxiv.org/abs/2304.07193).
6. B. Cheng, I. Misra, A. G. Schwing, A. Kirillov and R. Girdhar, “Masked-attention Mask Transformer for Universal Image Segmentation,” CVPR, 2022. [Author manuscript](https://arxiv.org/abs/2112.01527).
7. E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv:2106.09685, 2021. [Author manuscript](https://arxiv.org/abs/2106.09685).
8. A. Francis et al., “Sentinel-2 Cloud Mask Catalogue,” Zenodo record 4172871, 2020. [Dataset record](https://doi.org/10.5281/zenodo.4172871). Dataset semantics are also checked against the locally retained original README.
9. CloudSEN12 dataset, Zenodo record 7320147. [Dataset record](https://doi.org/10.5281/zenodo.7320147). The actual source-training revision and split are pinned by the internal experiment manifest; this citation does not replace that pin.
10. T. Shtym et al., “KappaSet: Sentinel-2 KappaZeta Cloud and Cloud Shadow Masks,” Version 1, Zenodo record 7100327, 2022. [Dataset record](https://doi.org/10.5281/zenodo.7100327).

## Internal editorial note — remove from a submission copy

This version provides continuous text, available results and a linked bibliography. It remains an internal empirical report, not a certified submission package. The main missing evidence is independent confirmation and comparable strong adaptation baselines. A broader novelty review and intervention replication would also be needed if the claims were expanded. Figures, a venue-specific layout and public release URLs have not been fabricated to conceal these gaps. No additional experiment is automatically authorized by this draft.
