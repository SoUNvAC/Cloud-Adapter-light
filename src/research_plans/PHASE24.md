# Phase 24 — Hardware-aware static DINO block screen

## Dependency and motivation

Phase 24 starts only after the three-seed clean V12 baseline passes Phase 23.
The historical decoder ablations showed that query compression yields little
latency reduction, while almost all V12 parameters remain in frozen DINOv2-S.
This phase therefore screens whole-backbone structured compression.

The proposed mechanism replaces inactive DINO blocks with parameter-free
identities while retaining original block numbers for pretrained weight
mapping. The forward graph statically omits inactive blocks, so ONNX export does
not require dynamic routing, token selection, or custom operators.

## Screen

```bash
cd src
bash tools/run_phase24_block_screen_4090d.sh
```

Using only the Phase 23 seed-42 checkpoint and official validation split, test:

- 12 blocks: `[0,1,2,3,4,5,6,7,8,9,10,11]`;
- 10 blocks: `[0,2,3,4,5,6,8,9,10,11]`;
- 8 blocks: `[0,2,4,5,7,8,10,11]`;
- 6 blocks: `[0,2,5,8,10,11]`;
- 4 blocks: `[2,5,8,11]`.

All four adapter/output taps `[2,5,8,11]` remain active. No candidate is trained
in this screening phase. Test remains sealed.

## Pre-registered qualification gates

- The 12-block validation result must reproduce Phase 23 seed 42 within 0.05 mIoU.
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
