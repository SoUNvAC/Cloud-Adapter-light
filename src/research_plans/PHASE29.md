# Phase 29 — MobileNetV2 compact-backbone joint pilot

## Motivation

Phase 28 rejected ResNet-18 because its L8 Biome zero-shot mean missed both the
absolute external floor and the allowed drop from V8. No L8-driven tuning is
allowed. Phase 29 changes architecture to a natively mobile, depthwise-separable
MobileNetV2 with independent ImageNet pretraining while retaining the frozen V8
decoder design and training protocol.

## Fixed protocol

- Backbone: standard `MobileNetV2`, width 1.0, output stages `(1,2,4,6)` with
  channels `(24,32,96,320)`, fully fine-tuned at 0.1 times decoder LR.
- Initialization: official OpenMMLab ImageNet checkpoint, SHA-256
  `3b2dc3afee0b94e52b357a60851f1ac8ec95cf9318762e785812edf7f6736b14`.
- Decoder, losses, schedule, seed 42, and official validation selection match
  Phase 26. Train 40,000 iterations from the independent ImageNet start.
- After validation selection, benchmark native PyTorch FP16 on 4090D and run
  exactly one zero-shot L8 Biome evaluation with the already frozen `[0,3,2,1]`
  taxonomy mapping. CloudSEN internal test remains sealed.

## Pre-registered gates and stop-loss

The pilot passes only if every gate holds:

- CloudSEN official validation mIoU >=66.0;
- L8 Biome zero-shot mIoU >=35.0;
- L8 mIoU drop from the Phase 28 three-seed V8 mean <=6.5 points;
- native-FP16 speedup versus the Phase 26 remeasured V8 baseline >=2.0x;
- deployment parameters <=7.5M and all metrics/latencies finite.

If the pilot passes, Phase 30 freezes it and adds seeds 123/3407 for joint
internal-validation and external confirmation. If any gate fails, the mobile
replacement direction is closed at one seed; Phase 30 becomes the final frozen
evidence audit and does not unseal CloudSEN test for a failed candidate.

## One-click command

```bash
cd src
bash tools/run_phase29_mobilenetv2_oneclick_4090d.sh
```

The report is `work_dirs/phase29_mobilenetv2_pilot/PHASE29_REPORT.txt`. The
script never starts Phase 30.
