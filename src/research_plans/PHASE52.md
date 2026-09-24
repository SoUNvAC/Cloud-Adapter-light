# Phase 52 — sparse target-sensor MsRE feasibility

## Rationale and separation from the stopped route

Phase 45 established a 13.0963-point supervised target-domain oracle gap.
Phases 46–49 showed that output-space factorization and adjacent unsupervised
adaptation do not exploit that gap; Phase 50 showed that 1% target labels can
raise target mIoU by 5.5870 points but updating the shared model forgets 3.6823
source points. Phase 51 closed uncertainty calibration. Phase 52 is therefore
a new representation-level, sensor-incremental hypothesis, not a relaxation
of any Phase 45–51 gate.

The frozen Phase 22 DINOv2-S, source Cloud-Adapter and source segmentation head
are the immutable source path. A target sensor selects a separate residual
path containing vanilla MsRE at blocks `[2, 5, 8, 11]`, 16 tokens per block,
and a zero-initialized 4x4 logit delta. Only these target parameters are
optimized from the already frozen Phase 50 selection of 65/6,502 target-train
patches. Target-val may select among iterations 1k–4k; target-test and the
CloudSEN internal test remain sealed.

## Phase 52A preregistered stop-loss

Before training, implementation checks must prove that disabling the target
path is a code-level identity and is numerically equal to an independently
built source model within `1e-5` maximum absolute logit error (CUDA kernels are
not assumed bitwise deterministic), only target modules require gradients, the selected
set is still exactly 65 images selected without labels, and added trainable
parameters are at most 0.50M. The public MsRE arithmetic is retained for this
vanilla baseline; sensor metadata gating and cloud-aware token groups are not
introduced in the same experiment.

The fixed best target-val checkpoint must satisfy every gate:

- target mIoU at least 46.1241 (+3.0 over source-only);
- target weak-class mIoU at least 19.3059 (+4.0);
- target macro Boundary F1 at least 16.4763 (+3.0);
- source-val mIoU at least 73.93 (at most 0.05 forgetting);
- target-specific trainable parameters at most 0.50M;
- measured batch-1 512x512 FP16 latency overhead at most 20% versus the same
  frozen Phase 22 model on the same machine and timing protocol;
- all 535 source-val and 1,905 target-val images evaluated with finite metrics;
- target-test and CloudSEN internal test remain sealed.

Any failed gate stops Phase 52. No token-count, injection-position, learning
rate, iteration, or loss-weight adjustment is allowed. Only if every gate
passes may Phase 52B compare 16 versus 32 tokens and sparse positions; only a
passing Phase 52B may add sensor-metadata scale gates and cloud-aware tokens.

## Future gates (inactive until Phase 52A passes)

Metadata-conditioned scale weighting must beat ordinary MsRE by at least 1.0
target mIoU or 1.5 weak-class IoU at matched labels and seeds, without exceeding
0.60M added parameters or 25% latency overhead. Cloud-aware tokens and ordinal,
boundary, or consistency losses are evaluated one factor at a time. A failure
closes that factor rather than triggering a neighboring hyperparameter sweep.

## Phase 52A result and stop decision

The fixed run completed 4,000 iterations and selected iteration 1,000 on
target-val. With the target path disabled, the same checkpoint reproduced
source-val at 73.9800 mIoU, so source forgetting was effectively zero. The
target path used 380,577 trainable parameters and increased matched RTX 4090 D
FP16 batch-1 latency from 22.0689 ms to 24.4751 ms (+10.90%); both efficiency
gates passed.

On all 1,905 target-val images, the candidate reached 44.8525 mIoU, 15.2351
weak-class mIoU, and 16.5280 macro Boundary F1. Relative to the frozen
source-only baseline these are +1.7284, -0.0708, and +3.0517 points. The
boundary, source-preservation, parameter, latency, completeness, finite-value,
and seal gates passed, but the preregistered +3.0 target mIoU and +4.0 weak-IoU
gates failed. Cloud-shadow IoU was only 0.8403, despite thin-cloud IoU reaching
29.6299.

Phase 52 therefore executes `stop_phase52_msre_route`. The project must not
run the 32-token or injection-position ablations and must not add metadata
gating, cloud-aware tokens, ordinal losses, or consistency losses on top of
this candidate. Target-test and CloudSEN internal test remain sealed.

## Post-hoc protocol correction: official `Shadows?` field

The USGS L8 Biome catalogue contains a scene-level `Shadows? yes/no` field.
Only 32/96 scenes are `yes`; this field is not present in the downloaded MTL
files and was absent from the frozen Phase 45 manifest. Phase 52A therefore
incorrectly treated class-0 pixels in `Shadows?=no` scenes as verified
non-shadow negatives and evaluated shadow IoU on scenes without shadow truth.

The impact is material. Of the frozen 65 Phase 50 training patches, 12 come
from `yes` scenes and 53 from `no` scenes. The original 1,905-image target-val
contains 963 patches from eight `yes` scenes and 942 from eight `no` scenes.
Consequently, the published Phase 52A all-scene shadow IoU, weak-class mean,
macro Boundary F1 and the derived "shadow collapse" interpretation are marked
protocol-contaminated. They remain in the log for provenance but are not valid
evidence about shadow adaptation.

The frozen official-page snapshot is
`research_plans/protocol_data/l8_biome_usgs_shadow_status.csv`, SHA-256
`c735dc14ba5d40ffc0443ba9e5441d2716acd897c80f341a0093319f18862390`, with
source URL `https://landsat.usgs.gov/node/7` stored on every row.

## Phase 52A-fix preregistration

Phase 52A-fix changes only label validity. It keeps the same frozen Phase 22
checkpoint, the same 65-patch Phase 50 selection, 16 tokens, injection blocks
`[2, 5, 8, 11]`, rank-8 head delta, 380,577 trainable parameters, seed 52,
augmentations, optimizer, learning-rate schedule, batch size, 1,000-iteration
validation interval and 4,000 total iterations.

For a training patch from a USGS `Shadows?=no` scene, raw L8 target class 0 is
mapped to ignore (255), because it may be clear, fill or unlabelled shadow.
Thin- and thick-cloud pixels remain supervised. `Shadows?=yes` patches retain
the complete four-class label. Checkpoint selection and target reporting use
only the 963 patches from the eight `Shadows?=yes` target-val scenes. The
source-only baseline is recomputed on exactly those same 963 patches; the
target-test and CloudSEN internal test remain sealed.

The fixed candidate must pass every original relative gate on the corrected
subset: at least +3.0 mIoU, +4.0 thin/shadow mean IoU and +3.0 macro Boundary
F1 over corrected source-only. In addition, shadow IoU may not regress at all.
The source-val floor remains 73.93, target parameters must be exactly 380,577,
matched FP16 latency overhead must not exceed 20%, all 535 source and 963
corrected target samples must be evaluated with finite metrics, and both test
sets remain sealed. Any failure executes
`stop_phase52a_fix_shadow_status_route`; no post-hoc change to tokens,
positions, losses, sampling, learning rate or training length is permitted.
