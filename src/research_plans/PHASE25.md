# Phase 25 — Hardware-aware DINO MLP channel screen

## Dependency and motivation

Phase 24 showed that skipping whole DINO blocks destroys the continuous
representations used by Cloud-Adapter before it provides enough native speedup.
Phase 25 therefore retains all 12 attention/residual blocks and the complete V8
decoder, but narrows the dense hidden dimension of every DINO MLP. This is a
hardware-friendly structured width change: the resulting graph uses smaller
dense GEMMs and no masks, sparse kernels, dynamic routing, or custom operators.

## Deterministic initialization

Starting from the clean Phase 22 seed-42 V8 checkpoint, each MLP neuron receives
the path-strength score `||W_fc1[row]||_2 * ||W_fc2[column]||_2`. The top
channels are retained and restored in original index order. All other weights,
including attention, Cloud-Adapter, and deformable Mask2Former, are copied
exactly. Converted checkpoints contain model state and metadata only, so
artifact sizes are comparable and do not include optimizer state.

## Screen

Evaluate MLP ratios 4.0 (unpruned control), 3.0, 2.5, and 2.0 on the official
validation split. No candidate is trained. Benchmark native PyTorch FP16 on the
4090D with batch 1, 512x512 input, 20 warmups, and 100 measured iterations.
Test remains sealed.

## Pre-registered gates and stop-loss

- The converted 4.0 control must reproduce Phase 22 seed 42 within 0.05 mIoU.
- Every latency must be finite and positive.
- A narrowed candidate qualifies only if native-FP16 mean speedup is >=1.08x
  and zero-shot validation mIoU drop is <=5.0 points.
- If multiple candidates qualify, select the fastest; an exact latency tie
  selects the higher mIoU.

If no candidate qualifies, DINO MLP width pruning is closed and Phase 26 must
switch to an independently pretrained compact backbone. If a candidate
qualifies, Phase 26 freezes the ratio and performs paired three-seed
teacher-student fine-tuning. Neither threshold may be relaxed after results.

## One-click command

```bash
cd src
bash tools/run_phase25_mlp_screen_oneclick_4090d.sh
```

The report is written to
`work_dirs/phase25_mlp_screen/PHASE25_REPORT.txt`. The script never starts Phase
26.
