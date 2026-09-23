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
