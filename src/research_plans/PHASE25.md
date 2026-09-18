# Phase 25 — Weak-class and boundary-aware recovery for static pruning

## Method

Phase 25 fine-tunes the fastest Phase 24 candidate using the clean V8 checkpoint
with the same seed as a frozen online teacher. Unlike the failed Phase 10 global
KL loss, the proposed loss gives 2x weight to thin cloud and cloud shadow pixels
and an additional 3x total weight at label boundaries (base 1 plus boundary
weight 2). It does not match structurally incompatible intermediate features.

Teacher execution and all distillation losses are removed from the saved student
checkpoint. Inference retains only the statically pruned DINO backbone, standard
FPN, and compact query decoder.

## Training and automatic direction switch

```bash
cd src
bash tools/run_phase25_block_kd_3seed_4090d.sh
```

Qualified Phase 24 candidates are tried from fastest to slowest. Each uses seeds
42, 123, and 3407, the paired clean V12 checkpoint as initialization, and the
paired clean V8 checkpoint as teacher. If a candidate misses the pre-registered
gate, the script automatically switches to the next slower qualified candidate.
Test remains sealed.

## Pre-registered gates

- Every seed's validation mIoU drop from paired clean V12 must be <= 1.0 point.
- Mean paired validation mIoU drop must be <= 0.50 point.
- Three-seed mIoU standard deviation must be <= 0.60.
- Mean paired drop of `(thin-cloud IoU + cloud-shadow IoU) / 2` must be <= 0.50.
- Phase 24 Native-FP16 speedup must remain >= 1.15x.

## Stop-loss decision

If every qualified candidate fails, close static block pruning. Do not increase
KD weights or relax gates after observing validation. The next method direction
is compact-backbone replacement or width/channel distillation. Only a passing
candidate proceeds to ONNX export, real INT8, external datasets, and final test.
