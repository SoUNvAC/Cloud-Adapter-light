# Phase 64 RGB 现有结果终点审计

日期：2026-10-06
范围：只分析已有 checkpoint 与 target-val 产物；未启动训练，未读取 target-test，未按结果删除任何 scene。

## 1. 公平口径更正

早期表格用“旧四类 CloudSEN checkpoint 聚合三父类”作为 target source-only，而 MsRE/LoRA 实际从另训的三父类 source checkpoint 初始化。前者适合历史对照，但不能隔离适配增量。本审计统一使用同一个三父类 checkpoint：

| Target-val | 公平 source-only mIoU | Shadow IoU | MsRE 三 seed mIoU（均值±SD） | 公平 ΔmIoU | 公平 ΔShadow |
|---|---:|---:|---:|---:|---:|
| L8 | 58.0530 | 26.3248 | 61.5169±0.4877 | +3.4639 | +1.7768±1.1750 |
| SPARCS | 55.5775 | 20.3268 | 67.4076±0.4719 | +11.8301 | +21.8396±0.4805 |

该修正取代旧表中的 target delta，不改变模型绝对指标和源域开关结果。所有 target 训练都使用密集真值掩膜，应称为**监督式域适配**或目标域监督微调，不是无监督域适配。

## 2. L8 八个 scene 的三 seed 配对结果

`shadow→surface/cloud` 是 source-only 真值 Shadow 像元被预测为相应父类的比例；`Δ` 为 MsRE 三 seed 相对公平 source-only 的平均变化，单位为百分点（pp）。

| Scene | Biome | GT Shadow | ΔmIoU（均值±SD） | ΔShadow IoU（均值±SD） | source sh→surface | Δsh→surface | source sh→cloud | Δsh→cloud |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LC80010112014080LGN00 | snow/ice | 5.49% | +10.93±2.18 | +12.87±2.33 | 52.40% | -17.74 | 22.85% | -1.44 |
| LC80250022014232LGN00 | snow/ice | 2.77% | -1.04±2.76 | -2.29±1.76 | 25.30% | +18.37 | 20.61% | -15.16 |
| LC80320382013278LGN00 | shrubland | 3.44% | +1.99±0.56 | +1.54±0.94 | 33.77% | +2.11 | 9.80% | -5.40 |
| LC80980712014024LGN00 | grass/crops | 5.66% | +5.44±0.56 | +0.28±0.59 | 14.30% | +5.27 | 7.55% | -4.09 |
| LC81010142014189LGN00 | wetlands | 3.14% | +2.50±0.23 | -0.11±0.10 | 9.34% | +4.81 | 3.79% | -2.54 |
| LC81750622013304LGN00 | forest | 11.74% | +4.00±1.56 | +5.51±3.23 | 78.89% | -3.22 | 6.48% | -2.74 |
| LC81930452013126LGN01 | barren | 9.09% | +3.45±0.35 | +8.84±0.84 | 10.21% | +2.06 | 33.91% | -9.11 |
| LC82020522013141LGN01 | grass/crops | 1.11% | +7.13±1.00 | +1.24±0.33 | 37.76% | +15.44 | 37.39% | -22.55 |

观察（仅作假设生成）：

- 只有 `LC80250022014232LGN00` 的平均 `ΔmIoU` 与 `ΔShadow` 同时为负，但另一 snow/ice scene 是八个 scene 中改善最大者，因此不能用预定义 biome 或 snow/ice 条件解释损失，也不得据此剔除该 scene。
- `shadow→cloud` 在 8/8 scene 均下降；`shadow→surface` 却在 6/8 scene 上升。两项错误流向相反，表明适配普遍把 Shadow 判别边界从 Cloud 一侧推向 Surface 一侧，最终 Shadow 收益取决于两种变化的净效应。
- 可预注册、跨 scene 检验的候选假设是：**MsRE 的 Shadow 收益由适配前的 Shadow 混淆几何（`shadow→surface` 与 `shadow→cloud`）预测，而不是由 biome 名称预测。**但它是基于仅 8 个 scene 的事后诊断；当前没有独立 scene、没有预注册，也没有验证一个可部署的约束机制，不能作为已成立贡献。

## 3. 公平比较表

“开启源域”表示直接用目标适配分支评估 CloudSEN；“关闭恢复”表示已知输入域后关闭增量分支。关闭 MsRE/LoRA 后，预测 SHA256 与三父类 source-only 完全一致，共享参数和 buffer mismatch 均为 0。参数量为目标适配时可训练参数。

| Target / 方法 | target mIoU | 相对公平 source-only | CloudSEN：分支开启 | CloudSEN：关闭恢复 | 可训练参数 | 目标标注预算 |
|---|---:|---:|---:|---:|---:|---|
| L8 / source-only | 58.0530 | — | 77.8179 | 77.8179 | 0 | 0 |
| L8 / shared-parent | 64.10 | +6.05 | 63.21 | 未定义 | 1.317M | 1,560 个密集标注 patch，18 scenes |
| L8 / full FT | 51.51 | -6.54 | 38.21 | 不可关闭 | 23.610M | 同上 |
| L8 / LoRA | 62.80 | +4.75 | 66.6042 | 77.8179 | 0.295M | 同上 |
| L8 / MsRE seed64 | 61.0182 | +2.97 | 73.3432 | 77.8179 | 0.381M | 同上 |
| L8 / MsRE 3-seed mean | 61.5169 | +3.46 | — | 77.8179 | 0.381M | 同上 |
| SPARCS / source-only | 55.5775 | — | 77.8179 | 77.8179 | 0 | 0 |
| SPARCS / shared-parent | 69.65 | +14.07 | 69.92 | 未定义 | 1.317M | 60/60 个 target-train 密集标注 scenes |
| SPARCS / full FT | 61.05 | +5.47 | 50.54 | 不可关闭 | 23.610M | 同上 |
| SPARCS / LoRA | 68.82 | +13.24 | 68.1511 | 77.8179 | 0.295M | 同上 |
| SPARCS / MsRE seed64 | 67.1705 | +11.59 | 73.7240 | 77.8179 | 0.381M | 同上 |
| SPARCS / MsRE 3-seed mean | 67.4076 | +11.83 | — | 77.8179 | 0.381M | 同上 |

L8 的 1,560 个 patch 是 `Shadows?=yes` 完整标签 target-train 池的 100%，占冻结完整 target-train manifest 的 `1560/6502=23.99%`；SPARCS 使用预定 target-train split 的 60/60 个密集标注 scene。两者均不是“1% 标注”或 UDA 设置。跨 seed 共用一个 source checkpoint，因此只测目标适配随机性，不测源训练随机性。

## 4. 一页贡献审计

| 项目 | 来源/状态 | 本项目可以声称什么 | 不能声称什么 |
|---|---|---|---|
| learnable tokens、多尺度 depthwise conv、projector、稀疏残差注入 | 已有 MsRE 结构 | 将公开 MsRE 作为参数高效适配载体并复现实验 | 不能包装成本文新机制 |
| 三父类映射与 shared-parent 基线 | 本项目实现并验证 | 在两个目标域完成统一 coarse label space 的监督式比较 | 不能证明数据集特定细类解耦有效 |
| 增量分支显式开/关、共享状态与预测哈希审计 | 本项目实现并验证 | 已知域切换时，MsRE/LoRA 可精确恢复冻结源路径 | 这是可切换系统属性，不是同时兼顾源/目标，也不足以构成强算法创新 |
| target-head residual / adapter gate | 本项目代码机制，未独立消融 | 可作为实现细节披露 | 不能归因目标收益，不能宣称新贡献有效 |
| scene-disjoint 协议、三 seed、配对 scene/分层 bootstrap、LOSO | 本项目实现并验证 | 更严格地界定收益稳定性；SPARCS 稳定，L8 的 scene 不确定性仍在 | 不能把 L8 正点估计写成普遍稳定收益 |
| 层级概率一致性、数据集特定 child head | 原型/设想，未训练验证 | 无实证贡献 | 不能声称解决异构细粒度标签空间，不能用于论文标题或核心结论 |
| source forgetting 控制 | 关闭分支时成立；开启时仍下降 | 支持“已知域条件下可切换 PEFT” | 不支持“同一开启模型保持源性能”；L8/SPARCS MsRE 开启后分别下降 4.47/4.09 点 |

## 5. 项目决定

**结束当前 RGB + 三父类 + MsRE 组合的方法开发。**不进入 64C，不再通过增加 seed、模块或数据集数量寻找论文故事；保留 SPARCS 的稳定正结果，作为监督式 PEFT 与已知域切换的可靠基线。

理由：分析提出了“适配前 Shadow 混淆方向预测收益”的可检验诊断假设，但它来自 8-scene 事后观察，尚未转化为一个与已有 MsRE 不同、经跨 scene 预注册验证的机制。按照本轮终点规则，候选假设不足以推翻止损决定。若未来以独立项目重启，唯一合理入口是先在未见 scene 上预注册验证该预测关系，再决定是否研究 confusion-direction-constrained residual；不能使用当前 scene 反复调参。

## 6. 可追溯性

- 三父类 source checkpoint SHA256：`64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`
- `three_parent_target_baseline.json`：`587108e401fb2509f0432e4aeb78fd6e7667b472e73034cb94e98378a60a9ad6`
- `l8_scene_analysis.json`：`5da9dae8d3fcb429a6bc25a638e509a3c6cbab479e2b93796f420d0a70798b2b`
- `l8_scene_analysis.csv`：`d33275e1fdd7880e918d67d58476ab63a188ffec8c035dce4ca221654fc3fe51`
- target-test 读取：`false`
