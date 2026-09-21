# Phase 34 — weak-class and boundary-aware KD pilot

## Motivation

Phase 33 failed the stability gate and its 66.7133 mean remains 7.09 points
below the clean V8 anchor. Further compression is therefore stopped. This phase
tests a single accuracy-recovery mechanism that does not alter the deployed
Phase 31 graph: training-only semantic distillation from the clean V8 teacher,
weighted toward thin cloud, shadow, and label boundaries.

## Fixed protocol

- Student: Phase 31 MobileNetV2 + LiteFPN seed-42 best checkpoint, warm-started.
- Teacher: frozen Phase 22 V8 seed-42 best checkpoint, online FP16 inference.
- Loss: temperature-2 semantic KL, weight 1.0; class weights
  `[1.0, 1.0, 2.5, 2.0]`; boundary multiplier 1.5 at radius 1.
- Batch size 1, base learning rate 1e-5, 10,000 iterations, validation every
  1,000 iterations, deterministic seed 42.
- Teacher parameters are excluded from student checkpoints. Deployment is
  benchmarked with the plain Phase 31 inference config.
- L8 and CloudSEN internal test are not read.

## Pre-registered gates and stop-loss

- official validation mIoU >=68.0;
- improvement over the paired Phase 31 seed-42 run >=1.5 points;
- mean IoU of thin cloud and cloud shadow >=51.0;
- native-FP16 speedup versus V8 >=2.0x and parameters <=3.0M;
- all metrics finite and internal test remains sealed.

If every gate passes, Phase 35 performs one frozen L8 zero-shot evaluation. If
any gate fails, this KD recipe is closed without changing its loss weights,
temperature, duration, or threshold; the next direction must change the
student architecture or training supervision rather than tune against L8.

## One-click command

```bash
cd src
bash tools/run_phase34_litefpn_kd_oneclick_4090d.sh
```

The report is `work_dirs/phase34_litefpn_kd/PHASE34_REPORT.txt`.
