# Phase 61 实验记录

## 2026-09-24：协议冻结与启动

- 目标：审计 CloudSEN12/L8 Biome 的标签血缘、优化充分性和标签粒度；准备独立双盲人工复核。
- 改动：新增 61A 血缘表、61B 固定 4000-step 浅层优化、61C 同预测标签折叠、61D 盲化复核包及评分脚本。
- 网络/读出：不训练分割网络。61B 只在冻结 Phase52 `pixel_decoder_mask` 上训练线性 softmax readout；61C 只重算既有 confusion；61D 只做冻结推理和 RGB crop。
- 数据口径：L8 Biome 仅 `Shadows?=yes` 可作为完整四类标签；Phase52A-fix 的 53 张 `no` 图块仅保留 thin/thick 有效像元，raw target ID 0 为 ignore。
- 止损线：61A 任一名称集合/hash/路径/mask/mapping 血缘不清即暂停 61B–D；61D 在两位独立人员完成前不得报告 κ，也不得发表“标签体系不一致”的核心结论。
- 结果：尚未产生；不得提前填写。

## 2026-09-25：61A 协议血缘审计完成

- 目标：核实 Phase50/52A-fix/55/56/60 是否共享 manifest、选择集、类别映射与 checkpoint，并定位 53 张样本的排除原因。
- 改动：无训练；逐项读取 manifest、selection、USGS metadata、mask 和既有 cache metadata，生成血缘 JSON/Markdown 表。
- 网络/读出：Phase50 使用 source checkpoint `d33a81337544…`；Phase52A-fix 及后续 adapted 分支使用 `9655ee83c638…`；manifest hash 均为 `885c2d7f61ca…`。
- 止损线：任一名称集合/hash/路径/mask合法值/映射不一致即暂停后续。
- 结果：全部硬门通过。Phase50 对 6502 张 RGB 打分并选同一组 65 张，不读取标签；Phase52A-fix 实际读取这 65 张，其中 53 张 `Shadows?=no` 的 raw target ID 0 被 ignore，仍有 9,987,543 个有效监督像元；Phase55/56/60A 则整图过滤这 53 张，读取同一组合法 12 张（3,145,728 像元）。
- 排除原因：53/53 均仅因 scene 级 `Shadows?=no`；不存在路径缺失、mask 非法值、类别映射或 split 错误，也没有阶段暗换另一组 65 张。
- 产物：lineage JSON SHA256 `d7a5a6317a173977085c4f4fc719a091649886f4755d26c693a582c57547016b`；表 SHA256 `0d52b386643a417ef2cf99da0811ebcbfc946604b0df926dbfe03fb113c763bf`。判定 `lineage_clear_continue_61b_61c`。
