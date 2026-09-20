# Phase 24 — Hardware-aware static DINO block screen

## Dependency and motivation

Phase 24 starts after Phase 23 closes the standard-FPN branch. It returns to the
passed three-seed Phase 22 V8 accuracy anchor and keeps its deformable decoder
unchanged. The historical decoder ablations and Phase 23 failure show that
replacing the decoder costs too much accuracy, while most V8 parameters and
latency remain in frozen DINOv2-S. This phase therefore screens whole-backbone
structured compression.

The proposed mechanism replaces inactive DINO blocks with parameter-free
identities while retaining original block numbers for pretrained weight
mapping. The forward graph statically omits inactive blocks, so ONNX export does
not require dynamic routing, token selection, or custom operators.

## Screen

```bash
cd src
bash tools/run_phase24_block_screen_4090d.sh
```

Using only the Phase 22 seed-42 checkpoint and official validation split, test:

- 12 blocks: `[0,1,2,3,4,5,6,7,8,9,10,11]`;
- 10 blocks: `[0,2,3,4,5,6,8,9,10,11]`;
- 8 blocks: `[0,2,4,5,7,8,10,11]`;
- 6 blocks: `[0,2,5,8,10,11]`;
- 4 blocks: `[2,5,8,11]`.

All four adapter/output taps `[2,5,8,11]` remain active. No candidate is trained
in this screening phase. Test remains sealed.

## Pre-registered qualification gates

- The 12-block validation result must reproduce Phase 22 seed 42 within 0.05 mIoU.
- Benchmark latencies must be finite and positive.
- A pruned candidate qualifies for fine-tuning only if:
  - Native FP16 mean speedup is >= 1.15x; and
  - zero-shot validation mIoU drop is <= 8.0 points.

The 8-point zero-shot allowance is a screening threshold, not a final accuracy
claim. Final candidates must recover accuracy in paired three-seed training.

## Stop-loss decision

If no candidate qualifies, static DINO depth pruning is closed and the project
switches to width/channel distillation or a compact pretrained backbone. If one
or more candidates qualify, Phase 25 fine-tunes the fastest qualifying candidate
with a class- and boundary-aware teacher objective and uses paired three-seed
statistics. Thresholds are not relaxed after observing Phase 24.

## Recorded outcome

The 12-block anchor reproduced 73.98 mIoU. The 10/8/6/4-block candidates
obtained 56.17/40.41/34.94/13.47 mIoU and 1.047/1.102/1.169/1.234x native-FP16
speedups. No candidate met both gates. Static block skipping is closed; Phase 25
must not fine-tune these candidates and instead switches to width/channel
distillation or an independently pretrained compact backbone. Test remained
sealed.
