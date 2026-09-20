# Phase 26 — ImageNet-pretrained ResNet-18 compact-backbone pilot

## Motivation

Phases 24 and 25 show that post-hoc removal of DINOv2-S blocks or MLP channels
causes catastrophic representation loss before producing useful end-to-end
speedup. Phase 26 switches direction rather than relaxing either stop-loss. It
uses a natively compact hierarchical ResNet-18 backbone with independent
ImageNet pretraining and retains the proven V8 128-channel, 25-query,
two-layer pixel/query Mask2Former decoder.

## Fixed protocol

- Backbone: standard `ResNetV1c-18`, outputs at strides 4/8/16/32, fully
  fine-tuned; backbone learning rate is 0.1 times the decoder learning rate.
- Initialization: official OpenMMLab ImageNet checkpoint
  `resnet18_v1c-b5776b93.pth`, SHA-256
  `b5776b937a850b71ede79225265fabb5665b4641008e9b840929a94a919de707`.
- Decoder: Phase 22 V8 design, randomly initialized; no DINO or Phase 22 model
  weights are transferred.
- Pilot seed: 42, 40,000 iterations, validation every 2,000 iterations and best
  checkpoint selected only on official validation.
- Deployment benchmark: native PyTorch FP16, batch 1, 512x512, 20 warmups and
  100 measured iterations on the same 4090D. Phase 22 seed-42 V8 is benchmarked
  again in the same run. Test remains sealed.

## Pre-registered gates and stop-loss

The pilot passes only if all of the following hold:

- validation mIoU is at least 68.0;
- mean native-FP16 speedup versus the remeasured V8 baseline is at least 1.25x;
- the deployed graph has at most 18.0 million parameters;
- all latency measurements are finite and positive.

If the pilot passes, Phase 27 freezes this configuration and runs seeds 123 and
3407 for a three-seed confirmation (seed 42 is reused). If any gate fails,
ResNet-18 is stopped after one seed and Phase 27 changes to a pretrained mobile
backbone rather than tuning this pilot or relaxing thresholds.

## One-click command

```bash
cd src
bash tools/run_phase26_resnet18_oneclick_4090d.sh
```

The combined report is written to
`work_dirs/phase26_resnet18_pilot/PHASE26_REPORT.txt`. The script never starts
Phase 27.
