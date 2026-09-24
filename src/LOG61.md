# Phase 61 实验记录

## 2026-09-24：协议冻结与启动

- 目标：审计 CloudSEN12/L8 Biome 的标签血缘、优化充分性和标签粒度；准备独立双盲人工复核。
- 改动：新增 61A 血缘表、61B 固定 4000-step 浅层优化、61C 同预测标签折叠、61D 盲化复核包及评分脚本。
- 网络/读出：不训练分割网络。61B 只在冻结 Phase52 `pixel_decoder_mask` 上训练线性 softmax readout；61C 只重算既有 confusion；61D 只做冻结推理和 RGB crop。
- 数据口径：L8 Biome 仅 `Shadows?=yes` 可作为完整四类标签；Phase52A-fix 的 53 张 `no` 图块仅保留 thin/thick 有效像元，raw target ID 0 为 ignore。
- 止损线：61A 任一名称集合/hash/路径/mask/mapping 血缘不清即暂停 61B–D；61D 在两位独立人员完成前不得报告 κ，也不得发表“标签体系不一致”的核心结论。
- 结果：尚未产生；不得提前填写。
