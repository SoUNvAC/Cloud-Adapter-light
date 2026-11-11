# Phase 29 preregistration: fair PEFT baselines

Phase 29 tests whether Cloud-Adapter contributes beyond a frozen foundation
model and a competitive parameter-efficient adaptation method. Historical
comparisons with different backbone sizes, decoders, splits, or training budgets
are not accepted as this evidence.

## Frozen comparison

Three methods use the same pretrained DINOv2-S, identical 512 x 512 inputs,
identical standard-op V12 LiteFPN plus compact Mask2Former head, the official
clean train/validation split, FP32 training, batch 2, 40,000 iterations, and
seeds 42, 123, and 3407:

1. `frozen`: backbone frozen; only the V12 head is trained.
2. `rein`: LoRA-REIN adaptation in all 12 DINO blocks; the same head is trained.
3. `cloud_adapter`: Cloud-Adapter interactions; the same head is trained.

All heads start randomly initialized. Phase 22 or Phase 23 checkpoints must not
be loaded. Before training, `audit_phase29_fair_baselines.py` builds all models
and requires identical decoder hashes, schedule hashes, head parameter counts,
clean split paths, and fresh initialization. The REIN and Cloud-Adapter
backbone-trainable parameter counts must be within 2.0x; otherwise token/rank
dimensions are adjusted using parameter counts only, before any accuracy run.

Run:

```bash
bash tools/run_phase29_fair_baselines_4090d.sh
```

The CloudSEN12 test split remains sealed. Each best checkpoint is evaluated by
redirecting the test loop to official validation.

## Hard gates and stop-loss

All nine runs must finish and satisfy:

1. Every run validation mIoU >= 50.0 and each method's three-seed standard
   deviation <= 0.75.
2. Cloud-Adapter mean paired mIoU gain over the frozen-backbone baseline >= 1.0
   percentage point.
3. Cloud-Adapter mean paired weak-class mIoU gain (thin cloud and cloud shadow)
   over the frozen baseline >= 1.0 point.
4. Cloud-Adapter mean paired mIoU difference from parameter-matched REIN >=
   -0.50 point.

Failure means the evidence does not support a Cloud-Adapter-specific advantage.
The compression pipeline must then adopt the strongest fair PEFT baseline or
redesign the adaptation module; it may not continue to frame ordinary frozen
backbone behavior as the proposed method's contribution. Results are appended
to `EXPERIMENT_LOG.md` only after all runs complete.
