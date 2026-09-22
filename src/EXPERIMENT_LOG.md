# Cloud-Adapter 轻量化与边缘部署实验记录

> 本文件是 Phase 实验的唯一持续记录。自 Phase 16 起，每完成一个 Phase，继续在本文件末尾追加，不另建实验记录文件。

- 最后更新：2026-09-10
- 研究目标：在尽量保持 CloudSEN12 High L1C 云检测精度的前提下，降低 Cloud-Adapter 的参数存储、推理延迟和部署算子复杂度，使其更适合星载、无人机和其他资源受限的边缘设备。
- 主要硬件：NVIDIA GeForce RTX 4090 D 24 GB。
- 默认任务：CloudSEN12 High L1C 四类语义分割，类别为 `clear`、`thick cloud`、`thin cloud`、`cloud shadow`。
- 默认输入：`512 × 512`，部署基准默认 `batch size = 1`。
- 主干路线：冻结 DINOv2-S，训练 Cloud-Adapter 与分割解码器；Control 仅用于诊断解码器精度上限。
- 数值约定：分割指标均以百分数记录。没有从日志或实验输出中得到的数据统一标为“未记录”，不进行推测。
- 指标约定：现有各次输出中的 mFscore 与 mDice 数值相同，因此明细表省略重复的 mFscore 列；Phase 1 保留该列用于说明原始输出格式。

## 总览

| Phase | 候选模型/产物 | 主要动作 | mIoU | Mean latency | Throughput | Artifact | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | V1-LightHead | 建立 DINOv2-S 轻量基线 | 64.68 | 12.057 ms | 82.94 img/s | 7.47 MiB | 速度快，但精度不足 |
| 2 | Control-Mask2Former | 恢复完整 Mask2Former 诊断上限 | 72.18 | 39.106 ms | 25.57 img/s | 187.10 MiB | 精度上限，不能作为轻量部署模型 |
| 2 | V2-WeightedFusion | 拼接 PMAA 特征并学习尺度权重 | 63.51 | 未记录 | 未记录 | 未记录 | 失败，弱类精度下降 |
| 3 | V3-LightFPN | 轻量自顶向下 FPN | 64.83 | 未记录 | 未记录 | 未记录 | 比 V1 略好，仍远低于 Control |
| 4 | V4-Concat | 多尺度通道拼接后融合 | 64.27 | 未记录 | 未记录 | 未记录 | 失败，拼接没有恢复语义能力 |
| 5 | V5-Q50-D3-P3 | 结构化压缩 Mask2Former | 71.62 | 25.165 ms | 39.74 img/s | 104.86 MiB | 成功，接近 Control |
| 6 | V6-Q25-D3-P3 | 查询数 50 → 25 | 71.56 | 25.039 ms | 39.94 img/s | 104.85 MiB | 几乎无精度损失 |
| 7 | V7-Q25-D2-P3 | 查询解码器 3 → 2 层 | 71.44 | 23.631 ms | 42.32 img/s | 101.74 MiB | 有效的小幅提速 |
| 8 | V8-Q25-D2-P2 | 像素编码器 3 → 2 层 | 71.28 | 21.595 ms¹ | 46.31 img/s¹ | 45.16 MiB¹ | 选为高精度轻量基线 |
| 9 | V9-Micro | 特征宽度 128 → 64 | 70.10 | 22.911 ms | 43.65 img/s | 97.48 MiB | 精度损失大于速度收益 |
| 10 | V8-Logit-KD | Control → V8 语义 logit 蒸馏 | 71.10 best | 未记录 | 未记录 | 约 93 MB | 未超过原始 V8 |
| 11 | V8-FP16/W8-Sim | FP16 存储与 W8 敏感性筛选 | 71.29 best | 21.595 ms¹ | 46.31 img/s¹ | 45.17 MiB | FP16 无损；模拟 W8 基本无损 |
| 12 | V12-ExportFPN | 可导出的标准算子 FPN 替换可变形像素编码器 | 68.73 | 17.147 ms | 58.32 img/s | 44.31 MiB | 建立高速、可导出档位 |
| 13 | V12-Feature-KD | V8 → V12 像素特征蒸馏 | 67.59 | 未记录 | 未记录 | 未记录 | 失败，低于未蒸馏 V12 |
| 14 | V12-ONNX | 静态 ONNX FP16 图导出 | 未单独重测 | 未测试 | 未测试 | 未记录 | 导出及结构检查通过 |
| 15 | V12-ONNXRuntime | ORT CUDA 数值与性能验收 | 68.728² | 10.056 ms | 99.44 img/s | 沿用 Phase 14 | 通过，默认可移植部署后端 |
| 16-A | V12-TensorRT-FP16 | FP16 ONNX 直接构建 TensorRT 10.0.1 | 无效 | 2.834 ms³ | 352.86 img/s³ | 53.34 MiB | 失败，输出含 NaN |
| 16-B2 | V12-TensorRT-Mixed | FP32 ONNX 加敏感层 FP32 约束 | 无效 | 未测试 | 未测试 | 49.92 MiB | 失败，18/20 图全部输出非有限值 |
| 16-C | V12-TensorRT-FP32 | 严格 FP32 TensorRT 诊断 | 68.728² | 5.926 ms | 168.75 img/s | 95.54 MiB | 通过，固定 NVIDIA 环境的极速后端 |
| 16-D | V12-TensorRT-BF16 | FP32 ONNX 启用 BF16、关闭 FP16/TF32 | 无效 | 未测试 | 未测试 | 56.17 MiB | 失败，17/20 图全部输出非有限值 |
| 16-E | V12-TensorRT-MixBF16 | BF16 主计算加敏感路径 FP32 | 无效 | 未测试 | 未测试 | 56.19 MiB | 失败，19/20 图全部输出非有限值；关闭低精度 TRT 路线 |
| 17 | V12-FullDeployment | 975 图三后端完整精度复核 | 68.721 / 68.728 / 68.728⁴ | 未测试 | 未测试 | 沿用现有产物 | 通过，部署转换无可测精度损失 |
| 18 | V12-EndToEnd | ORT/TRT CPU 张量至 CPU 掩膜端到端基准 | 沿用 Phase 17 | 9.565 / 6.652 ms⁵ | 104.55 / 150.32 img/s⁵ | 44.75 / 95.54 MiB⁵ | 通过，固定 NVIDIA 平台首选 TensorRT |
| 19 | V12-ResolutionPareto | 512/448/320 输入分辨率完整精度与端到端筛选 | 68.728 / 68.211 / 65.215⁶ | 8.804 / 9.894 / 8.247 ms⁶ | 113.58 / 101.07 / 121.26 img/s⁶ | 44.75 / 45.47 / 45.42 MiB⁶ | 失败，固定回512输入 |
| 20 | V12-StandaloneMask | ONNX 图内 ArgMax/Cast，移除运行时 PyTorch 后处理 | 68.728 | 13.942 ms⁷ | 71.73 img/s⁷ | 44.75 MiB | 通过，最终采用纯 session.run |
| 21 | Clean-Protocol | 恢复官方 train/val/test，执行全量哈希与类别分布审计 | 不训练 | 不适用 | 不适用 | Manifest | 通过；Phase 22 起禁止用 test 选模 |

¹ Phase 8 此处采用 Phase 11 中同机测得的 Native-FP16 结果；早期 autocast 基准为 23.448 ms、42.65 img/s、101.03 MiB。

² Phase 15/16 当时只完成20图 parity；表中 ORT 与 TensorRT-FP32 的完整集 mIoU 来自 Phase 17 的975图独立测量。

³ Phase 16-A 的 TensorRT 输出包含非有限值，数值验收失败；该延迟和吞吐只能说明错误图执行速度，不能作为有效部署性能或论文结果。

⁴ Phase 17 总览行中的三个 mIoU 依次对应 PyTorch-FP16、ONNXRuntime-CUDA、TensorRT-FP32。

⁵ Phase 18 总览行中的两项依次对应 ONNXRuntime-CUDA、TensorRT-FP32，计时为 CPU 张量到 CPU 掩膜的端到端口径，不能与此前 GPU 常驻输入的内核口径直接混用。

⁶ Phase 19 总览行中的三项依次对应512、448、320输入；所有预测均恢复为512×512掩膜后评价或计时。

⁷ Phase 20 对比后保留更快的 Trial A 纯 `session.run` 结果；预分配 OrtValue I/O binding 的 Trial B 为14.960 ms、66.85 img/s，未被采用。

## Phase 1 — V1-LightHead 轻量基线

### 简介

用 DINOv2-S 替换大型 VFM，并设计不含可变形注意力和查询解码器的 `LightCloudHead`，作为整条轻量化路线的第一条速度基线。

### 目标

- 验证冻结 DINOv2-S 加四处 Cloud-Adapter 交互能否在 24 GB 单卡上稳定训练。
- 将解码器缩减为 64 通道的投影、均值多尺度融合和深度可分离卷积。
- 建立后续精度、延迟和模型体积的比较起点。

### 网络与训练改动

- DINOv2-S：12 个 Transformer block，交互/输出位置为 `[2, 5, 8, 11]`。
- Cloud-Adapter：4 层，`context_dim=64`、`hidden_channels=64`、`rank_dim=8`。
- 解码器：四尺度投影到 64 通道，resize 到最高分辨率后取均值，接两个深度可分离残差块。
- 损失：交叉熵加 Dice，权重均为 1.0。
- 训练：AMP、batch 4、40,000 iter、每 2,000 iter 验证。
- 配置：`configs/light/cloud_adapter_dinov2_s_light_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mFscore | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 86.26 | 64.68 | 74.89 | 77.08 | 77.08 | 80.07 | 74.89 |

四类 IoU：clear 84.55、thick cloud 80.31、thin cloud 43.86、cloud shadow 49.99。

部署基准：22.403 M 参数、7.47 MiB checkpoint、12.057 ms、82.94 img/s、峰值显存 0.317 GiB。

### 结论

速度非常好，但 thin cloud 和 cloud shadow 明显偏低。需要用完整 Mask2Former 做同主干对照，判断损失主要来自 VFM 缩小还是解码器过度简化。

## Phase 2 — Control 上限诊断与 V2 加权融合

### 简介

Phase 2 同时包含两个互补实验：Control 在完全相同的 DINOv2-S 和 Cloud-Adapter 上恢复完整 Mask2Former；V2 则保持轻量头，加入 PMAA 特征拼接和可学习尺度权重。

### 目标

- 用 Control 定位 Phase 1 的精度瓶颈。
- 验证局部卷积缓存和动态尺度加权能否弥补轻量头的边界与弱类信息损失。

### 网络与训练改动

- Control：256 通道、100 queries、6 层多尺度可变形像素编码器、9 层查询 Transformer 解码器。
- Control 因 Hungarian matching 的 FP16 稳定性问题，使用 FP32、batch 1。
- V2：主干开启 `has_cat=True`，四尺度输入通道由 384 变为 448；融合方式改为可学习 `weighted_sum`。
- 配置：`configs/control/cloud_adapter_dinov2_s_mask2former_4x_l1c.py` 与 `configs/light/cloud_adapter_dinov2_s_light_v2_l1c.py`。

### 实验结果

| Model | aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control-Mask2Former | 89.25 | 72.18 | 83.24 | 82.99 | 82.88 | 83.24 |
| V2-WeightedFusion | 85.89 | 63.51 | 73.57 | 76.00 | 79.76 | 73.57 |

| Model | Clear IoU | Thick IoU | Thin IoU | Shadow IoU |
| --- | ---: | ---: | ---: | ---: |
| Control-Mask2Former | 87.87 | 84.79 | 53.35 | 62.71 |
| V2-WeightedFusion | 84.18 | 79.92 | 39.96 | 49.97 |

Control 部署基准：42.200 M 参数、187.10 MiB、39.106 ms、25.57 img/s、峰值显存 0.433 GiB。

### 结论

Control 比 V1 高 7.50 mIoU，证明主要瓶颈是过度简化的分割头，而不是 DINOv2-S 本身。V2 比 V1 反而下降 1.17 mIoU，简单拼接局部缓存和尺度加权不能代替强像素/查询建模。

## Phase 3 — V3 轻量自顶向下 FPN

### 简介

在保持 V1 主干和适配器完全不变的条件下，将一次性 resize-and-average 融合改为逐级自顶向下 FPN。

### 目标

保留多尺度层级关系，在不引入 Mask2Former 的情况下改善 thin cloud 和 cloud shadow。

### 网络与训练改动

- 四尺度侧向投影到 64 通道。
- 从最深特征开始逐级上采样、相加，并用深度可分离残差块细化。
- 仍采用交叉熵加 Dice、AMP、batch 4、40,000 iter。
- 配置：`configs/light/cloud_adapter_dinov2_s_light_v3_fpn_l1c.py`。
- `fpn_aux` 深监督配置已建立，但当前记录中没有对应完成结果，不纳入结论。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 86.48 | 64.83 | 74.63 | 77.16 | 80.93 | 74.63 |

四类 IoU：clear 84.74、thick cloud 80.58、thin cloud 42.85、cloud shadow 51.15。

### 结论

仅比 V1 提升 0.15 mIoU。FPN 改善了 shadow，但 thin cloud 下降，仍无法解决轻量卷积头缺少全局查询建模的问题。

## Phase 4 — V4 多尺度拼接融合

### 简介

对 V1 做单变量融合消融：四尺度投影后不再求均值，而是在通道维拼接，再用 1×1 投影融合。

### 目标

检验均值融合是否因丢失尺度身份而成为主要精度瓶颈。

### 网络与训练改动

- 主干、Cloud-Adapter、64 通道宽度、细化块、损失和训练日程均与 V1 相同。
- 唯一核心变化是 `fusion="concat"`。
- 配置：`configs/light/cloud_adapter_dinov2_s_light_v4_concat_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 86.26 | 64.27 | 73.99 | 76.70 | 80.89 | 73.99 |

四类 IoU：clear 84.43、thick cloud 80.34、thin cloud 42.66、cloud shadow 49.65。

### 结论

比 V1 下降 0.41 mIoU。均值融合不是主要矛盾，继续堆叠轻量卷积融合的收益有限，应转向结构化压缩 Mask2Former。

## Phase 5 — V5 结构化压缩 Mask2Former

### 简介

从 72.18 mIoU 的 Control 出发，对 Mask2Former 进行成组结构压缩，而不是继续修补 LightCloudHead。

### 目标

保留像素解码器和查询解码器的核心表达能力，同时显著缩减宽度、查询数和层数。

### 网络与训练改动

- 特征/输出通道：256 → 128。
- Queries：100 → 50。
- 像素编码器：6 → 3 层。
- 查询 Transformer：9 → 3 层。
- Attention heads：8 → 4；FFN 宽度相应缩至 512。
- FP32 训练，batch 2，其他主干、适配器、数据、损失和 40,000 iter 日程与 Control 一致。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_lite_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 89.03 | 71.62 | 82.51 | 82.58 | 82.75 | 82.51 |

四类 IoU：clear 87.69、thick cloud 84.21、thin cloud 52.68、cloud shadow 61.88。

部署基准：24.065 M 参数、104.86 MiB、25.165 ms、39.74 img/s、峰值显存 0.349 GiB。

### 结论

只比 Control 低 0.56 mIoU，却将延迟降低约 35.65%，说明 Mask2Former 存在大量结构冗余。V5 成为后续逐变量压缩的起点。

## Phase 6 — V6 查询数减半

### 简介

对 V5 做单变量消融，将对象查询数从 50 减至 25。

### 目标

确认四类云分割是否需要大量 queries，并减少查询侧计算和存储。

### 网络与训练改动

- 仅将 `num_queries=50` 改为 `25`。
- 128 通道、3 层像素编码器和3层查询解码器保持不变。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 89.10 | 71.56 | 82.06 | 82.52 | 83.04 | 82.06 |

四类 IoU：clear 87.75、thick cloud 84.35、thin cloud 52.42、cloud shadow 61.71。

部署基准：24.059 M 参数、104.85 MiB、25.039 ms、39.94 img/s、峰值显存 0.313 GiB。

### 结论

相对 V5 仅下降 0.06 mIoU，25 queries 足以覆盖该四类任务。速度提升很小，说明当前延迟主要不在 query 数，而在主干和像素编码过程。

## Phase 7 — V7 查询解码器减层

### 简介

保持 V6 的通道宽度、25 queries 和像素编码器不变，将查询 Transformer 解码器由3层减为2层。

### 目标

继续削减查询侧计算，寻找精度和延迟的平衡点。

### 网络与训练改动

- 唯一核心变化：`transformer_decoder.num_layers: 3 → 2`。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 89.02 | 71.44 | 82.11 | 82.44 | 82.83 | 82.11 |

四类 IoU：clear 87.66、thick cloud 84.26、thin cloud 52.45、cloud shadow 61.40。

部署基准：23.794 M 参数、101.74 MiB、23.631 ms、42.32 img/s、峰值显存 0.323 GiB。

### 结论

相对 V6 下降 0.12 mIoU，同时平均延迟下降约 5.62%。这是有效的结构压缩。

## Phase 8 — V8 像素编码器减层

### 简介

保持 V7 的 128 通道、25 queries 和2层查询解码器，将多尺度可变形像素编码器由3层减为2层。

### 目标

直接压缩高分辨率空间注意力计算，并选定后续蒸馏、量化和部署的高精度学生模型。

### 网络与训练改动

- 唯一核心变化：`pixel_decoder.encoder.num_layers: 3 → 2`。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 88.86 | 71.28 | 82.42 | 82.33 | 82.36 | 82.42 |

四类 IoU：clear 87.55、thick cloud 84.08、thin cloud 52.30、cloud shadow 61.19。

早期 autocast 基准：23.610 M 参数、101.03 MiB、23.448 ms、42.65 img/s、峰值显存 0.234 GiB。

Phase 11 同机 Native-FP16 基准：45.16 MiB、21.595 ms、46.31 img/s、峰值显存 0.280 GiB。

### 结论

相对 V7 下降 0.16 mIoU，仍保持 71.28。V8 被选为高精度轻量基线及后续压缩教师/学生枢纽。

## Phase 9 — V9-Micro 通道减半

### 简介

在 V8 上继续压缩 Mask2Former 的表示宽度，将相关注意力、FFN 和位置编码尺寸同步缩小。

### 目标

测试 64 通道的微型查询解码器能否以很小的精度代价换取进一步速度收益。

### 网络与训练改动

- 特征/输出通道：128 → 64。
- FFN：512 → 256。
- 位置编码特征数：64 → 32。
- 25 queries、2层查询解码器、2层像素编码器保持不变。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_micro_q25_d2_p2_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 88.35 | 70.10 | 81.55 | 81.44 | 81.44 | 81.55 |

四类 IoU：clear 87.18、thick cloud 83.32、thin cloud 50.99、cloud shadow 58.93。

部署基准：22.686 M 参数、97.48 MiB、22.911 ms、43.65 img/s、峰值显存 0.324 GiB。

### 结论

相对 V8 损失 1.18 mIoU，而早期同口径延迟只改善约 2.29%。宽度继续减半不是划算的 Pareto 方向，停止沿 V9 继续压缩。

## Phase 10 — Control 到 V8 的语义 Logit 蒸馏

### 简介

以 72.18 mIoU 的 Control 为冻结在线教师，从 V8 最佳 checkpoint 开始进行短程语义 logit 蒸馏。

### 目标

尝试在不改变 V8 推理结构和 checkpoint 体积的情况下，恢复结构压缩丢失的约 0.9 mIoU。

### 网络与训练改动

- 教师：Control-Mask2Former，FP16、无梯度，仅训练时存在，不写入学生 checkpoint。
- 蒸馏：逐像素 KL divergence，temperature 2.0、distill weight 2.0。
- 学生：从 V8 最佳 checkpoint 热启动。
- 训练：batch 1、基础学习率 `2e-5`、10,000 iter、每 1,000 iter 验证。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_v8_distill_l1c.py`。

### 实验结果

- 最佳 checkpoint：iter 9,000，mIoU 71.10。
- 训练结束验证：mIoU 70.96。
- 最佳值演进：70.56@1k、70.88@2k、71.06@6k、71.07@8k、71.10@9k。
- 最佳 checkpoint 文件约 93 MB。

最终验证汇总：

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 88.72 | 70.96 | 82.02 | 82.08 | 82.25 | 82.02 |

最终四类 IoU：clear 87.22、thick cloud 84.14、thin cloud 51.59、cloud shadow 60.91。

### 结论

最佳值仍比原始 V8 低 0.18 mIoU，普通语义 logit KD 没有带来增益。该方向关闭，不使用 Phase 10 权重继续部署。

## Phase 11 — FP16 与 W8 量化敏感性筛选

### 简介

不立即引入真实 INT8 引擎，先对 V8 进行 FP16 权重导出和逐输出通道对称 W8 fake-quant 筛选，判断量化造成的精度风险。

### 目标

- 验证 FP16 权重存储是否无损。
- 比较只量化 VFM 与量化全部二维以上权重的敏感性。
- 建立 Native-FP16 的真实速度、显存和 checkpoint 基线。

### 量化方式

- FP16：所有浮点权重以 FP16 存储。
- W8-Sim：权重按输出通道映射到 `[-127, 127]`，再反量化并以 FP16 保存。
- W8-Sim 仅用于精度筛选，不是实际 INT8 checkpoint，也不会产生 INT8 内核加速。
- 工具：`tools/export_quant_screen_checkpoint.py`。

### 实验结果

| Artifact | Size (MiB) | aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V8-Original | 101.03 | 88.86 | 71.28 | 82.42 | 82.33 | 82.36 | 82.42 |
| V8-FP16-Weights | 45.16 | 88.86 | 71.28 | 82.41 | 82.33 | 82.36 | 82.41 |
| V8-W8-VFM-Sim | 45.17 | 88.86 | 71.27 | 82.41 | 82.32 | 82.34 | 82.41 |
| V8-W8-All-Sim | 45.17 | 88.86 | 71.29 | 82.47 | 82.33 | 82.32 | 82.47 |

| Artifact | Mean | P50 | P90 | Throughput | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| V8-Autocast | 24.015 ms | 23.851 ms | 24.659 ms | 41.64 img/s | 0.309 GiB |
| V8-Native-FP16 | 22.008 ms | 21.904 ms | 22.740 ms | 45.44 img/s | 0.280 GiB |

另一次与 Phase 12 同机基准中的 V8 Native-FP16 为 21.595 ms、46.31 img/s。

### 结论

FP16 将 checkpoint 缩小约 55.3%，完整测试集 mIoU 不变，同时降低延迟和显存。W8 fake-quant 的精度几乎不受影响，说明权重量化有潜力；但真实 INT8 是否加速仍需 TensorRT/QDQ 或其他实际量化后端验证。

## Phase 12 — V12 标准算子 ExportFPN

### 简介

保留 V8 已验证的 128 通道、25 queries 和2层查询 Transformer，将不利于通用部署的多尺度可变形像素编码器替换为仅含标准算子的轻量 FPN 像素解码器。

### 目标

- 去除可变形注意力及其自定义算子依赖。
- 降低实际推理延迟，而不仅是减少参数量。
- 为 ONNX Runtime 和 TensorRT 建立可直接编译的部署图。

### 网络与训练改动

- 新增 `LiteFPNPixelDecoder`。
- 四尺度先做 1×1 Conv + GroupNorm，再逐级上采样相加。
- 每级使用深度可分离卷积细化，只包含 Conv、GroupNorm、ReLU、Add 和 Resize 等标准运算。
- 保留 V8 的查询 embeddings 和2层查询解码器。
- 从 V8 最佳 checkpoint 热启动；新 FPN 从头初始化。
- 基础学习率 `2e-5`，FPN 学习率倍率 5，20,000 iter、每 1,000 iter 验证。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 87.64 | 68.73 | 80.47 | 80.40 | 80.58 | 80.47 |

四类 IoU：clear 86.45、thick cloud 82.24、thin cloud 49.57、cloud shadow 56.66。

同机 Native-FP16 部署基准：

| Model | Params | Checkpoint | Mean | P50 | P90 | Throughput | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V8-Native-FP16 | 23.610 M | 45.16 MiB | 21.595 ms | 21.412 ms | 22.289 ms | 46.31 img/s | 0.280 GiB |
| V12-ExportFPN-FP16 | 23.167 M | 44.31 MiB | 17.147 ms | 17.102 ms | 17.615 ms | 58.32 img/s | 0.279 GiB |

### 结论

相对 V8，V12 损失 2.55 mIoU，但延迟降低 20.60%，吞吐提升 25.93%，并消除了可变形注意力部署障碍。它不是高精度档位的替代品，而是新的高速、标准算子 Pareto 档位。

## Phase 13 — V8 到 V12 的像素特征蒸馏

### 简介

使用 V8 的可变形像素解码器作为训练期教师，蒸馏 V12 标准 FPN 的 mask feature 和三层 pyramid feature。

### 目标

在不改变 V12 推理图和速度的前提下，恢复标准 FPN 替换造成的 2.55 mIoU 损失。

### 网络与训练改动

- 教师：V8，FP16、冻结、仅运行主干和像素解码器。
- 学生：从 Phase 12 最佳 checkpoint 热启动。
- 特征损失：通道维 cosine loss。
- 公共蒸馏权重 0.5；mask 与 pyramid 分量权重均为 1.0。
- 关闭语义 logit KD，避免和特征目标混杂。
- batch 1、基础学习率 `1e-5`、学生 FPN 学习率倍率 5、10,000 iter。
- 配置：`configs/light/cloud_adapter_dinov2_s_mask2former_export_fpn_distill_l1c.py`。

### 实验结果

| aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 87.08 | 67.59 | 79.50 | 79.51 | 79.97 | 79.50 |

四类 IoU：clear 85.99、thick cloud 81.34、thin cloud 48.11、cloud shadow 54.90。

### 结论

比未蒸馏 Phase 12 再下降 1.14 mIoU。可变形像素特征与标准 FPN 表征存在较强结构不匹配，直接 cosine feature matching 会约束学生学习。Phase 13 权重不进入部署链。

## Phase 14 — V12 静态 ONNX 导出

### 简介

将未蒸馏的 Phase 12 模型导出为固定 batch、固定分辨率的标准域 ONNX 图。

### 目标

- 获得不依赖 MMEngine/MMSeg 和自定义 MMCV 算子的部署中间格式。
- 将预处理与 Mask2Former 语义 logits 合成逻辑固化进图。
- 为 ORT CUDA 和 TensorRT 提供同一个源模型。

### 导出改动

- 导出时禁用 xFormers，确保注意力展开为可移植运算。
- 输入契约：`float32[1,3,512,512]`，RGB，数值范围 0–255。
- 图内完成 mean/std 标准化，并转为 FP16 计算。
- 输出契约：`[1,4,512,512]` semantic logits。
- 固定 shape，ONNX opset 17，不设置 dynamic axes。
- 导出器先验证部署 wrapper 与 MMSeg `decode_head.predict` 的 PyTorch 一致性。
- 导出后运行 ONNX checker，并拒绝 `mmcv`、`xformers`、`ATen` 等非标准 operator domain。
- 工具：`tools/export_phase12_onnx.py` 与 `tools/export_phase14_onnx_4090d.sh`。

### 实验结果

- 导出流程完成，说明 wrapper parity、ONNX checker 和标准 operator domain 检查均未触发失败门槛。
- 产物：`work_dirs/phase14_v12_onnx/v12_export_fpn_fp16_512.onnx`。
- 本次对话没有保存导出脚本打印的 ONNX 文件大小和 wrapper 误差数值，标记为未记录。
- 本 Phase 不训练，也没有独立 mIoU 或速度结果。

### 结论

V12 已从研究框架模型转化为静态标准 ONNX 图，可以进入真实推理后端验收。

## Phase 15 — ONNX Runtime CUDA 一致性与性能验收

### 简介

使用 ONNX Runtime GPU 后端运行 Phase 14 产物，并与禁用 xFormers 后的 PyTorch FP16 wrapper 做真实图像数值对齐和同口径延迟测试。

### 目标

- 检查 ONNX 执行结果是否保持语义预测。
- 确认 Conv、MatMul、Softmax、Resize 等核心计算没有回退 CPU。
- 使用 CUDA I/O binding 排除不必要的 CPU↔GPU 输入输出拷贝，测量 GPU 常驻输入延迟。

### 实验设置

- 后端：ONNX Runtime GPU 1.18.0，CUDA 12/cuDNN 8 构建。
- 样本：20 张 CloudSEN12 High L1C 真实测试图。
- 基准：warmup 20、measurement 100、batch 1、512×512、FP16 compute。
- 比较对象：PyTorch-FP16 与 ONNXRuntime-CUDA。
- 验收门槛：像素 argmax agreement ≥99.9%，且核心计算算子不得回退 CPU。
- 工具：`tools/validate_phase12_onnxruntime.py` 与 `tools/validate_phase15_onnxruntime_4090d.sh`。

### 实验结果

数值一致性：

| Samples | Max abs error | Mean abs error | Argmax agreement |
| ---: | ---: | ---: | ---: |
| 20 | 0.0234375 | 0.000679097 | 99.95144% |

执行节点分配：

- CUDAExecutionProvider：20,020 次节点事件。
- CPUExecutionProvider：5,480 次节点事件。
- CPU 算子类型：Concat、Div、Gather、Mul、Slice、Split、Squeeze、Unsqueeze。
- Conv、MatMul、Einsum、Softmax、Resize、LayerNormalization 等核心算子没有回退 CPU。

同进程部署基准：

| Backend | Mean | P50 | P90 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch-FP16 | 17.366 ms | 16.970 ms | 18.197 ms | 57.58 img/s |
| ONNXRuntime-CUDA | 10.056 ms | 9.955 ms | 10.596 ms | 99.44 img/s |

ORT 相对 PyTorch：1.727× 等效吞吐加速，平均延迟降低 42.09%。部署门槛输出为 `PASSED`。

### 结论

Phase 15 成功。CPU 节点均为形状/索引类辅助路径，实际延迟没有显示出严重同步瓶颈。ORT 版本已成为当前 V12 的首选部署后端。该20图 parity 测试足以验证图转换，但论文最终结果仍应补充完整 975 图的后端 mIoU 复测。

该结果随后进入 Phase 16 的 TensorRT 评估；只有在保持至少 99.9% 像素决策一致性的同时，相对 ORT 获得足够大的额外加速，才值得承担 TensorRT 引擎与硬件/版本绑定的部署复杂度。

## Phase 16 — TensorRT 多精度引擎构建与验收

### 简介

将 Phase 14 的静态 ONNX 编译为 NVIDIA TensorRT 10.0.1 引擎，目标是在 4090D 上进一步融合算子和选择高效 kernel，同时保持 Phase 12 的语义输出。

### 目标

- TensorRT 对 PyTorch 的像素 argmax agreement 不低于 99.9%。
- TensorRT 输出不得包含 NaN 或 Inf。
- 只有数值验收通过后才比较速度；若 TensorRT 相对同次 ORT 没有足够收益，则仍优先使用可维护性更高的 ORT。
- 保存 TensorRT 版本、GPU、compute capability、I/O 和 engine 大小等元数据。

### Trial A：FP16 ONNX 直接构建 FP16 Engine

#### 网络与部署改动

- 输入：Phase 14 的 FP16 内部计算 ONNX。
- Builder：TensorRT 10.0.1，启用全局 FP16，4 GiB workspace。
- 固定输入：batch 1、RGB float32、512×512。
- 数值检查：20 张真实测试图；性能检查：warmup 20、measurement 100。

#### 实验结果

Engine 大小：53.34 MiB。

| Comparison | Max abs error | Mean abs error | Argmax agreement |
| --- | ---: | ---: | ---: |
| TensorRT vs PyTorch | 0.0629883 | NaN | 58.24303% |
| TensorRT vs ONNX Runtime | 0.0629883 | NaN | 58.24759% |

| Backend | Mean | P50 | P90 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch-FP16 | 17.465 ms | 17.437 ms | 17.701 ms | 57.26 img/s |
| ONNXRuntime-CUDA | 7.374 ms | 7.292 ms | 7.565 ms | 135.62 img/s |
| TensorRT-FP16 | 2.834 ms | 2.813 ms | 2.917 ms | 352.86 img/s |

#### 结论

Trial A 失败。`mean_abs=NaN` 表明 TensorRT 输出含非有限值，约 58% 的 argmax agreement 很可能是 NaN 传播后预测退化造成的偶然类别重合。虽然执行速度极快，但该速度对应错误输出，不能进入 Pareto 表、论文结论或实际部署。

初步原因是 FP16 源图把大量中间张量类型提前固定为半精度，TensorRT 难以为归约、归一化、除法、幂和 Softmax 等敏感路径自动提升精度。FP16 的有限动态范围可能产生 Inf，随后传播为 NaN。

### Trial B：FP32 源图加受控混合 FP16

#### 修正目标

- 从 Phase 12 checkpoint 重新导出 FP32 ONNX，给 TensorRT 留出混合精度选择空间。
- 卷积和矩阵计算仍允许 FP16；Normalization、Reduce、Softmax、Div、Pow、Sqrt 等敏感层强制 FP32。
- Builder 使用 `OBEY_PRECISION_CONSTRAINTS`，不能满足约束时直接构建失败。
- 验证器新增 PyTorch、ORT、TensorRT 三方 NaN/Inf 精确计数，并优先执行非有限值门槛。
- 如果受控混合 FP16 仍失败，再用严格 FP32 TensorRT engine 进行诊断，区分解析/执行错误和低精度溢出。

#### Trial B1 结果

- PyTorch deployment wrapper parity：max abs 0、mean abs 0。
- ONNX checker：通过，opset 17，共 2,237 个标准域节点。
- 输入：`float32[1,3,512,512]` RGB 0–255。
- 输出：`[1,4,512,512]` semantic logits。
- FP32 ONNX 大小：88.99 MiB。
- 产物：`work_dirs/phase16_v12_tensorrt/v12_export_fpn_fp32_512.onnx`。
- TensorRT engine 未生成，未执行数值或性能测试。

Trial B1 在设置敏感层精度时中止：TensorRT 10.0.1 的 `network.get_layer()` 返回通用 `ILayer`，该绑定没有元素运算子类的 `.op` 属性。此错误属于 builder API 兼容性问题，与 FP32 ONNX 内容和模型数值无关。

#### Trial B2 结果

不再访问 `.op`，改用 TensorRT 已公开的 `layer.type` 和 ONNX parser 保留的层名识别 Div、Pow、Sqrt、Reciprocal 等敏感运算，成功生成49.92 MiB 的受控混合 FP16 engine。

| Comparison | Max finite abs error | Finite mean abs error | Argmax agreement | Reference non-finite | TensorRT non-finite |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorRT vs PyTorch | 0.0622559 | 0.000388520 | 58.24286% | 0 | 18,874,368 |
| TensorRT vs ONNX Runtime | 0.0561523 | 0.000386562 | 58.24745% | 0 | 18,874,368 |

`18,874,368 = 18 × 4 × 512 × 512`，说明20张验证图中恰有18张的完整四通道输出全部为 NaN/Inf。验证器在数值门槛处终止，没有执行性能测试。敏感层 FP32 约束仍不足以阻止非有限值传播，Trial B2 失败。

运行日志还报告 `enqueueV3()` 使用默认 CUDA stream。该问题只会引入额外同步和影响计时，不解释整图非有限值；验证器已改为专用非默认 stream，并显式建立输入/输出 stream 依赖。

### Trial C：严格 FP32 TensorRT 诊断

#### 目标

- 从同一份已通过 checker 的 FP32 ONNX 构建关闭 FP16 和 TF32 的严格 FP32 engine。
- 如果严格 FP32 输出有限且 agreement 通过，证明失败根因是 TensorRT 低精度路径，再继续评估 BF16 或更细粒度的混合精度。
- 如果严格 FP32 仍产生非有限值，则优先检查 TensorRT 10.0.1 对图中算子的解析/优化，而不是继续调整 FP16 层约束。

#### 实验结果

Engine 大小：95.54 MiB。

| Comparison | Max abs error | Mean abs error | Argmax agreement | Reference non-finite | TensorRT non-finite |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorRT vs PyTorch | 0.0323588 | 0.000652939 | 99.94823% | 0 | 0 |
| TensorRT vs ONNX Runtime | 0.0412255 | 0.000830729 | 99.94169% | 0 | 0 |

| Backend | Mean | P50 | P90 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch-FP16 | 17.958 ms | 17.578 ms | 18.813 ms | 55.68 img/s |
| ONNXRuntime-CUDA | 6.762 ms | 6.654 ms | 6.980 ms | 147.88 img/s |
| TensorRT-FP32 | 5.926 ms | 5.891 ms | 5.957 ms | 168.75 img/s |

#### 结论

Trial C 通过数值与部署门槛。TensorRT-FP32 相对同次 PyTorch 快 3.030×，相对 ORT 快 1.141×；平均延迟比 ORT 低 12.36%，吞吐高 14.11%。这证明 ONNX 图解析和 TensorRT 执行路径本身正确，Trial A/B2 的非有限值由低精度计算触发。

严格 FP32 engine 已经是有效候选，但95.54 MiB 体积约为 V12 FP16 checkpoint 的2.16倍，且只比 ORT 快约12%，部署收益仍不够理想，需要继续测试具有 FP32 动态范围的 BF16。

### Trial D：BF16 TensorRT

#### 目标

- 使用同一份 FP32 ONNX，仅允许 BF16 与必要的 FP32 fallback，不启用 FP16或TF32。
- 利用 BF16 与 FP32 相同的指数范围规避 FP16 overflow，同时争取接近 FP16 Tensor Core 的速度和较小 engine。
- 保持非有限值为0、对 PyTorch argmax agreement ≥99.9%。
- 若 BF16 通过且明显快于 FP32，则进入完整975图后端 mIoU 复测；否则保留 FP32 TensorRT 与 ORT 两个有效候选。

#### 实验结果

Engine 大小：56.17 MiB。

| Comparison | Max finite abs error | Finite mean abs error | Argmax agreement | Reference non-finite | TensorRT non-finite |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorRT vs PyTorch | 0.0957946 | 0.000804986 | 60.21481% | 0 | 17,825,792 |
| TensorRT vs ONNX Runtime | 0.114349 | 0.000821976 | 60.21931% | 0 | 17,825,792 |

`17,825,792 = 17 × 4 × 512 × 512`，说明20张验证图中有17张完整输出为 NaN/Inf。数值门槛在性能测试前终止，因此没有有效 BF16 延迟或吞吐结果。

#### 结论

Trial D 失败。BF16 虽然具有与 FP32 相同的指数范围，但较低的尾数精度仍可能使归一化、方差、除法或其他敏感链产生非法中间值；当前结果证明仅启用全局 BF16 不足以保证该模型稳定。严格 FP32 TensorRT 仍是唯一通过的 TensorRT 候选。

### Trial E：BF16 主计算加敏感路径 FP32

#### 目标与止损线

- 在 BF16 builder 上增加与 Trial B2 相同的 FP32 敏感层强约束，保留卷积和矩阵计算的 BF16 优化机会。
- 非有限值必须为0，对 PyTorch argmax agreement 必须达到99.9%。
- 若 Trial E 仍失败，停止继续调整低精度 TensorRT，直接保留 TensorRT-FP32 与 ORT 两条有效部署路径。
- 若 Trial E 通过，再根据同次延迟、engine 大小和完整975图 mIoU 决定最终后端。

#### 实验结果

Engine 大小：56.19 MiB。

| Comparison | Max finite abs error | Finite mean abs error | Argmax agreement | Reference non-finite | TensorRT non-finite |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorRT vs PyTorch | 0.0742188 | 0.00171671 | 59.32570% | 0 | 19,922,944 |
| TensorRT vs ONNX Runtime | 0.0893555 | 0.00177697 | 59.32972% | 0 | 19,922,944 |

`19,922,944 = 19 × 4 × 512 × 512`，说明20张图中有19张完整输出为 NaN/Inf。验证器在数值门槛终止，没有产生有效性能结果。

#### 结论

Trial E 失败并触发止损线。为 BF16 增加敏感层 FP32 约束仍不能消除非有限值，而且失败样本由17张增至19张。Phase 16 不再继续尝试低精度 TensorRT；有效部署候选固定为 ONNX Runtime CUDA 和严格 FP32 TensorRT。Phase 16 至此结束。

## Phase 17 — 完整测试集部署精度复核

### 简介

此前 Phase 15/16 只在20张真实图像上验证部署后端数值一致性。Phase 17 将在全部975张 CloudSEN12 High L1C 测试图上同时执行 PyTorch-FP16、ONNX Runtime CUDA 和 TensorRT-FP32，并直接根据标注计算完整分割指标。

### 目标

- 重新确认 PyTorch deployment wrapper 能复现约68.73 mIoU 的 Phase 12 结果。
- 分别计算 ORT 与 TensorRT-FP32 的 aAcc、mIoU、mAcc、mDice、mPrecision、mRecall 和四类别 IoU。
- 统计三后端全量像素 argmax agreement 与非有限值数量。
- ORT/TRT 对 PyTorch agreement 均需≥99.9%，mIoU 绝对差均需≤0.05个百分点，且所有后端非有限值为0。

### 实验设置

- 数据：`data/cloudsen12_high_l1c/{img_dir,ann_dir}/test` 全部975张图。
- 输入：RGB float32、batch 1、512×512；图内完成标准化。
- PyTorch：Phase 12 checkpoint，FP16 wrapper。
- ORT：Phase 14 FP16 ONNX，CUDAExecutionProvider，禁止 CPU fallback。
- TensorRT：Phase 16 Trial C 严格 FP32 engine。
- 工具：`tools/eval_phase17_full_deployment.py` 与 `tools/run_phase17_full_eval_4090d.sh`。

### 实验结果

完整975图分割指标：

| Backend | aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch-FP16 | 87.637 | 68.721 | 80.459 | 80.395 | 80.585 | 80.459 |
| ONNXRuntime-CUDA | 87.642 | 68.728 | 80.458 | 80.400 | 80.595 | 80.458 |
| TensorRT-FP32 | 87.639 | 68.728 | 80.473 | 80.400 | 80.582 | 80.473 |

逐类别 IoU：

| Class | PyTorch-FP16 | ONNXRuntime-CUDA | TensorRT-FP32 |
| --- | ---: | ---: | ---: |
| clear | 86.444 | 86.448 | 86.446 |
| thick cloud | 82.234 | 82.240 | 82.236 |
| thin cloud | 49.555 | 49.567 | 49.567 |
| cloud shadow | 56.653 | 56.657 | 56.663 |

全量有效像素一致性：

| Comparison | Argmax agreement |
| --- | ---: |
| PyTorch / ONNX Runtime | 99.94419% |
| PyTorch / TensorRT | 99.93800% |
| ONNX Runtime / TensorRT | 99.93201% |

三个后端的非有限 logits 数量均为0，验证器输出 `Full deployment accuracy gate: PASSED`。

### 结论

Phase 17 通过。PyTorch wrapper 得到68.721 mIoU，与 Phase 12 日志的68.73仅差0.009个百分点；ORT 和 TensorRT-FP32 均为68.728，相对 PyTorch 只高0.007个百分点。所有类别的后端间 IoU 差均不超过0.012个百分点，证明 ONNX 与 TensorRT 转换没有可测的任务精度损失。

最终保留两个部署档位：Phase 14 FP16 ONNX 加 ORT CUDA 作为默认可移植后端；严格 FP32 TensorRT 作为固定 NVIDIA GPU、追求最大吞吐的后端。依据 Phase 16-C 的同进程测试，TensorRT-FP32 相对 ORT 仅快1.141倍，但 engine 为95.54 MiB并绑定 TensorRT、GPU 架构和构建环境，因此不取代 ORT 成为默认交付格式。低精度 TensorRT 路线保持关闭。

## Phase 18 — 真实端到端部署口径基准

### 简介

Phase 15/16 的计时使用 GPU 常驻输入，主要反映后端图执行速度；边缘应用实际还需要把 CPU 图像张量送入 GPU，并把语义 logits 转为分割掩膜后取回 CPU。Phase 18 对两个通过完整精度验收的后端补充更接近在线处理链路的端到端计时。

### 目标

- 在同一进程、同一批真实图像和相同输出契约下公平比较 ORT CUDA 与 TensorRT-FP32。
- 计时范围固定为预处理后的 CPU `float32` RGB 张量，经 H2D、推理、GPU argmax，最终得到 CPU `uint8` 分割掩膜。
- 分别记录后端初始化时间、第一次推理延迟、稳态 Mean/P50/P90、吞吐、产物大小和增量 GPU 占用。
- 不把磁盘读取、图像解码和 resize 混入后端计时；这些共享前处理成本单独留给实际设备集成测试。
- 以端到端结果复核 Phase 17 的后端选择，并形成可直接迁移到 Jetson/无人机计算平台的基准脚本。

### 实验设置

- 硬件：NVIDIA GeForce RTX 4090 D 24 GB。
- 输入：20张真实测试图预先转换为 CPU `float32[1,3,512,512]`，batch 1。
- 输出：GPU 上完成四类 argmax，仅回传 CPU `uint8[1,512,512]` 掩膜。
- 基准：第一次调用单独计时，warmup 20，默认正式测量200次并轮换20张输入。
- ORT 使用 CUDA I/O binding、预分配 GPU 输出和 PyTorch CUDA stream；TensorRT 使用 Phase 16-C 严格 FP32 engine 与专用 CUDA stream。
- GPU 占用记录为后端创建前后 `cudaMemGetInfo` 的稳态增量，不等价于板端整机内存峰值。
- 工具：`tools/benchmark_phase18_e2e.py` 与 `tools/run_phase18_e2e_4090d.sh`。

### 实验结果

#### Trial A：ORT 符号输出维度导致工具中止

第一次运行在创建 ORT session 后、正式推理前中止。ONNX Runtime 将输出 batch 维报告为符号名 `Einsumseg_logits_dim_0`，而初版脚本直接对全部输出维度执行 `int()`，触发 `ValueError`。该错误不涉及模型数值、CUDA provider 或性能，未产生可用基准结果。

修复后，脚本使用已经在 Phase 14/15/17 验证的静态输出契约 `[1,4,512,512]` 分配 I/O binding 缓冲区，同时继续检查 ORT 元数据中所有明确给出的整数维度，避免掩盖真实 shape 不一致，并在 Trial B 重新运行。

#### Trial B：完整端到端基准

| Backend | Artifact | Startup | First | Mean | P50 | P90 | Throughput | GPU delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNXRuntime-CUDA | 44.75 MiB | 349.19 ms | 2196.381 ms | 9.565 ms | 9.500 ms | 10.709 ms | 104.55 img/s | 1144.0 MiB |
| TensorRT-FP32 | 95.54 MiB | 267.43 ms | 79.108 ms | 6.652 ms | 6.642 ms | 6.743 ms | 150.32 img/s | 338.0 MiB |

TensorRT 相对 ORT 的端到端平均延迟降低30.45%，等效吞吐提升1.438倍。TensorRT 的首帧延迟约为 ORT 的1/27.76，初始化时间降低约23.42%，稳态增量 GPU 占用降低806 MiB、约70.45%。代价是 engine 大小为 ORT ONNX 的约2.14倍。

两后端均完成201次实际掩膜回传，anti-lazy checksum 分别为38,920,114和38,948,620。checksum 只用于证明输出被实际消费，不作为一致性指标；二者在完整测试集上的正式一致性以 Phase 17 的99.93201% ORT/TensorRT 像素 agreement 为准。

### 结论

Phase 18 通过。固定 NVIDIA 环境的正式运行后端调整为 TensorRT-FP32：其30.45%的端到端延迟优势、显著更低的首帧等待和约70%的增量显存下降，已经足以抵消较大的 engine 文件。FP16 ONNX 仍是默认交换与可移植交付格式，并保留 ORT CUDA 作为无需目标机重建 engine 的通用运行后端。

这仍然只是4090D结果，不能直接宣称星载或无人机端实时。TensorRT engine 与 GPU 架构、TensorRT/CUDA 版本绑定，必须在最终 Jetson 或机载 NVIDIA 设备上由 FP32 ONNX 重新构建并复跑同一基准；当前4090D engine 不作为跨设备二进制发布物。

## Phase 19 — 输入分辨率精度—速度 Pareto 筛选

### 简介

Phase 18 已确定512×512下的有效部署后端，但边缘计算量仍受输入 token 数主导。Phase 19 不修改权重，增加448×448和320×320两个静态输入档位；按本项目 DINOv2-S 配置的 `patch_size=16`，512、448、320分别对应32×32、28×28、20×20 token 网格。同时三档均能被适配器/FPN的最大32倍下采样整除，不会产生不完整 patch 或非整数最深层尺度。所有预测 logits 均双线性恢复到512×512后再计算指标和输出掩膜。

### 目标

- 在完整975图上按同一 evaluator 独立测量512、448、320三档 ORT mIoU及逐类 IoU。
- 端到端计时继续采用 CPU float32 张量到 CPU 512×512 uint8 掩膜的 Phase 18 口径，并把低分辨率 logits 恢复到512的开销计入延迟。
- 以512档为直接基线；低分辨率候选需满足 mIoU 下降不超过0.50个百分点且端到端加速不少于1.15倍。
- 只对通过门槛的最快分辨率继续构建 TensorRT-FP32，避免为明显掉点的档位浪费 engine 调优时间。

### 网络与部署改动

- 模型参数、Phase 12 checkpoint、Cloud-Adapter和解码器完全不变。
- 从同一 checkpoint 分别导出固定448×448与320×320的 FP16 ONNX；保留 Phase 14 的512×512 ONNX作为基线。
- 推理输出先以双线性插值恢复至原始512×512空间，再执行四类 argmax。
- 工具：`tools/eval_phase19_resolution_pareto.py` 与 `tools/run_phase19_resolution_pareto_4090d.sh`。

### 实验设置

- 数据：CloudSEN12 High L1C 全部975张测试图。
- 后端：ONNX Runtime CUDA，CUDA I/O binding，禁止 CPU fallback。
- 精度：FP16 ONNX计算；输入契约仍为 RGB float32、0–255，图内完成归一化。
- 基准：20张真实图轮换，warmup 20、正式测量200次、batch 1；磁盘读取和图像解码不计时。
- 基线自检：512档 mIoU 必须落在 Phase 17 ORT 结果68.728±0.05内。
- 筛选门槛：相对512基线 mIoU drop ≤0.50，端到端 speedup ≥1.15×，且非有限 logits 为0。

### 实验结果

#### Trial A：位置编码插值的静态导出兼容性修复

首次导出低分辨率 ONNX 时在 DINOv2 位置编码插值处中止。PyTorch 2.1 ONNX tracer 将由输入 shape 推导的 `w0`、`h0` 保留为标量 Tensor，导致 `F.interpolate(scale_factor=(...))` 最终向 `upsample_bicubic2d` 传入 Tensor tuple，而该接口要求 Python float tuple，因此触发 `TypeError`。尚未进入 ORT 精度或性能测试。

修复将静态输入的目标 token 高宽显式转换为 Python `int`，再按原公式加0.1并计算 bicubic `scale_factor`。这不改变普通 eager 推理的数值公式，也不改权重；它仅把静态导出中的参数类型固定为 PyTorch 2.1算子接受的形式。修复后的完整筛选见 Trial B。

#### Trial B：完整分辨率 Pareto 筛选

完整975图精度：

| Input | aAcc | mIoU | Delta | mAcc | mDice | mPrecision | mRecall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 87.642 | 68.728 | +0.000 | 80.458 | 80.400 | 80.595 | 80.458 |
| 448 | 87.395 | 68.211 | -0.517 | 80.027 | 80.012 | 80.180 | 80.027 |
| 320 | 85.802 | 65.215 | -3.513 | 77.540 | 77.700 | 77.893 | 77.540 |

逐类别 IoU：

| Class | IoU@512 | IoU@448 | IoU@320 |
| --- | ---: | ---: | ---: |
| clear | 86.448 | 86.062 | 84.320 |
| thick cloud | 82.240 | 81.963 | 79.105 |
| thin cloud | 49.567 | 49.460 | 46.499 |
| cloud shadow | 56.657 | 55.360 | 50.937 |

ORT CUDA 端到端性能：

| Input | Tokens | Artifact | Startup | Mean | P50 | P90 | Throughput | Speedup | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 1024 | 44.75 MiB | 401.03 ms | 8.804 ms | 8.370 ms | 10.582 ms | 113.58 img/s | 1.000× | BASE |
| 448 | 784 | 45.47 MiB | 345.80 ms | 9.894 ms | 10.369 ms | 11.081 ms | 101.07 img/s | 0.890× | FAIL |
| 320 | 400 | 45.42 MiB | 336.32 ms | 8.247 ms | 7.494 ms | 10.117 ms | 121.26 img/s | 1.068× | FAIL |

三档均完成完整精度评估且没有触发非有限值门槛。anti-lazy checksum 分别为38,710,600、39,075,870、38,209,500，仅用于确认输出被实际消费。

### 结论

Phase 19 失败，没有低分辨率候选进入 TensorRT 构建。448档仅比精度门槛多下降0.017个百分点，但平均延迟反而增加12.38%，不构成 Pareto 改进；320档平均延迟只降低6.33%、加速1.068倍，却损失3.513 mIoU。

分辨率下降对小目标和边界敏感类别影响最明显。320档相对512的 clear、thick cloud、thin cloud、cloud shadow IoU 分别下降2.128、3.135、3.068、5.720个百分点。448档主要损失来自 cloud shadow（下降1.297），但其非理想 ORT kernel shape和恢复512分辨率的后处理成本抵消了 token 数下降。后续部署固定使用512×512，不再为448/320构建 TensorRT engine，也不采用低分辨率微调来挽救一个速度门槛已经失败的分支。

## Phase 20 — 无 PyTorch 的原生掩膜 ONNX

### 简介

Phase 18 的“CPU张量到CPU掩膜”计时仍依赖 PyTorch CUDA tensor完成 GPU argmax和内存桥接，不是独立的 ONNX Runtime 部署程序。Phase 20 将语义后处理固化进标准 ONNX 图，使产物直接返回四类 `uint8` 掩膜，并使用不导入 PyTorch、MMEngine、MMSeg或项目模型代码的验证器完成精度与性能验收。

### 目标

- 在 Phase 14 FP16 logits ONNX末尾追加标准域 `ArgMax(axis=1)` 和 `Cast(uint8)`，输出固定为 `uint8[1,512,512]`。
- 在20张真实图上与原始 logits ONNX的外部 argmax 达到至少99.99%像素一致率。
- 在完整975图上复现68.728±0.05 mIoU，并输出完整分割指标和逐类 IoU。
- 使用纯 NumPy/OpenCV/Pillow/ONNX Runtime链路测量 CPU RGB张量到CPU掩膜的启动、首帧和稳态延迟。
- 验证运行时无需 PyTorch；为后续 C++/Jetson 集成提供真正自包含的 ONNX语义输出契约。

### 网络与部署改动

- 主干、Cloud-Adapter、解码器、权重和512×512输入均不改变。
- 不重新训练或重新导出主计算图，只对已验收的 Phase 14 ONNX做确定性图后处理。
- logits `[1,4,512,512]` 经 `ArgMax(axis=1, keepdims=0)` 得到类别索引，再转为 `uint8`，显著缩小主机端输出张量。
- 工具：`tools/prepare_phase20_mask_onnx.py`、`tools/validate_phase20_standalone_ort.py` 与 `tools/run_phase20_standalone_ort_4090d.sh`。

### 实验设置

- 数据：CloudSEN12 High L1C 全部975张测试图；前20张同时用于 mask/logits图一致性。
- 后端：ONNX Runtime CUDA；CUDA provider必须为首选 provider并禁止运行时 fallback。
- 输入：CPU `float32[1,3,512,512]` RGB 0–255；输出：CPU `uint8[1,512,512]`。
- 基准：20张输入预加载，首次调用单独计时，warmup 20、正式测量200次；`session.run`计时包含输入H2D、图内argmax和掩膜D2H。
- 门槛：mIoU与 Phase 17 ORT相差≤0.05，mask/logits agreement ≥99.99%，输出dtype/shape严格匹配契约。

### 实验结果

#### Trial A：纯 `session.run` 独立部署

完整975图精度与 Phase 17 ORT logits图完全一致：

| Backend | aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORT-Standalone-Mask | 87.642 | 68.728 | 80.458 | 80.400 | 80.595 | 80.458 |

| Class | IoU |
| --- | ---: |
| clear | 86.448 |
| thick cloud | 82.240 |
| thin cloud | 49.567 |
| cloud shadow | 56.657 |

| Artifact | Startup | First | Mean | P50 | P90 | Throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 44.75 MiB | 312.99 ms | 1218.547 ms | 13.942 ms | 13.532 ms | 15.382 ms | 71.73 img/s |

mask图与原 logits图外部 argmax 的20图像素一致率为100.00000%，anti-lazy checksum为38,920,114；验证器确认未导入 PyTorch，输出 `Standalone mask deployment gate: PASSED`。

功能目标全部通过，但纯 `session.run` 相对 Phase 18 的 PyTorch tensor加 I/O binding ORT路径平均延迟增加45.76%，吞吐下降31.39%。此处只能确认差异来自两条 I/O、CUDA stream和同步路径的整体实现，不能在 Trial B 前单独归因于重复分配。首帧由2196.381 ms降至1218.547 ms，但仍有明显 CUDA kernel懒初始化成本。

#### Trial B：预分配 OrtValue I/O binding

在仍不导入 PyTorch的前提下，预先分配 CUDA input/output OrtValue并绑定固定地址；每帧通过 `OrtValue.update_inplace()`上传输入，图执行后只把 `uint8`掩膜下载为 NumPy。目标是消除 `session.run` 的重复分配开销，并尽量接近 Phase 18 的9.565 ms优化路径。

| I/O mode | Artifact | Startup | First | Mean | P50 | P90 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OrtValue-I/O-binding | 44.75 MiB | 297.62 ms | 1211.468 ms | 14.960 ms | 15.293 ms | 16.240 ms | 66.85 img/s |

完整精度仍为68.728 mIoU，mask/logits agreement仍为100.00000%，checksum仍为38,920,114，且运行时未导入 PyTorch；功能门槛再次通过。但是相对 Trial A，平均延迟增加1.018 ms、7.30%，吞吐下降6.80%。启动时间只降低4.91%，首帧只降低0.58%，不足以抵消稳态退化。Trial B性能优化失败。

### 结论

Phase 20 完成。原生掩膜 ONNX 在完整测试集上无精度损失，输出契约与独立运行时均通过，正式作为可移植交付产物。纯 ORT运行默认采用更快、更简单的 `session.run`，预分配 CUDA OrtValue不进入默认路径；工具保留 `--io-mode ortvalue-iobinding` 仅用于复现实验。

独立 ORT的13.942 ms仍明显慢于依赖 PyTorch CUDA tensor桥接的9.565 ms，也慢于 TensorRT-FP32的6.652 ms。因此运行方案分为三档：44.75 MiB mask ONNX加纯 ORT用于最少依赖和跨设备交付；logits ONNX加优化 I/O用于允许自定义设备内存桥接的 ORT集成；固定 NVIDIA设备继续优先采用目标机重建的 TensorRT-FP32。下一步不再在4090D上微调 Python I/O，转入真实目标设备复测。

## Phase 21 — 无泄漏评估协议与数据完整性审计

### 简介

Phase 1–20 的基础数据配置将 validation 和 test 同时指向 `img_dir/test`，多个训练配置又使用 validation mIoU 保存最佳 checkpoint。因此，历史测试集参与了 checkpoint、结构和部署候选选择。Phase 21 不删除或改写这些历史结果，而是将它们明确保留为开发期结果，并为 Phase 22 之后建立独立的官方 train/val/test 协议。

### 目标与止损线

- 恢复 CloudSEN12 High 原始的8,490张 train、535张 val、975张 test 三分割。
- 对全部图像和标注计算 SHA-256，任意两个 split 的精确图像重叠和图像-标注对重叠必须均为0。
- 所有图像必须存在同名标注，标签值只能为0、1、2、3。
- val/test 相对 train 的四类像素分布 total-variation distance 必须均不超过0.10。
- 任一硬门槛失败则停止后续训练，先重建数据；不得以随机 patch 切分规避场景泄漏问题。

### 协议与代码改动

- 保留 Phase 1–20 配置不变，以保证既有结果可复现。
- 新增 `configs/protocol/cloud_adapter_dinov2_s_mask2former_export_fpn_l1c_clean.py`，validation 使用 `img_dir/val`，test 保持 `img_dir/test`，并启用确定性训练设置。
- 修正 `tools/prepare_cloudsen12_l1c.py`，数据就绪检查现在要求 train、val、test 三分割全部存在。
- 新增 `tools/audit_cloudsen12_protocol.py`，生成逐样本哈希、标注直方图和完整 manifest。
- 一键入口：`bash tools/run_phase21_protocol_audit_4090d.sh`。本 Phase 不训练模型。

### 实验结果

数据归档 SHA-256 为 `a019db9779eda7080b60f2220c696747dbc26c469896e0fc4865d22b72c248de`，与下载源提供的校验值一致。

| Split | Samples | Clear | Thick cloud | Thin cloud | Cloud shadow | TV vs train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 8,490 | 54.649% | 26.810% | 9.678% | 8.863% | 0.0000 |
| Val | 535 | 53.592% | 26.831% | 10.775% | 8.802% | 0.0112 |
| Test | 975 | 52.535% | 28.997% | 8.955% | 9.514% | 0.0284 |

| Split pair | Exact image overlap | Exact image-label pair overlap |
| --- | ---: | ---: |
| Train / Val | 0 | 0 |
| Train / Test | 0 | 0 |
| Val / Test | 0 | 0 |

三个 split 的图像/标注数量完全配对，没有非法标签值，全部门槛通过。机器可读摘要保存于 `eval_result/phase21_protocol/result.json`；逐样本 manifest 默认写入被忽略的 `work_dirs/phase21_protocol/`，避免把约10,000条本地路径与哈希记录提交到仓库。

### 结论与后续约束

Phase 21 通过，说明官方三分割本身没有发现精确内容泄漏，原问题来自训练配置错误地用 test 代替 val。Phase 1–20 的数值继续作为开发历史保存，但不作为无偏最终测试证据。Phase 22 起，所有 checkpoint 和方向选择只能读取 val；test 只允许在结构、训练策略和阈值冻结后运行。

由于团队已经观察过历史 test 聚合结果，后续论文还必须增加此前未用于开发的外部数据集或新地域 holdout，作为真正的外部泛化证据；Phase 21 不能消除既往的人为测试集暴露。

## Phase 22 — 清洁协议下的三种子 V8 基线

### 简介

Phase 22 不提出新结构，而是在 Phase 21 的无泄漏协议下重新训练历史压缩候选 V8，建立 Phase 23 之后唯一允许使用的清洁验证基线。模型仍为 DINOv2-S、四处 Cloud-Adapter 交互和压缩 Mask2Former：128维特征、25个 query、2层 deformable pixel encoder、2层 query decoder。所有 checkpoint 选择和本 Phase 决策只读取官方 val；975张 test 在整个运行中保持封存。

### 目标与止损线

- 固定种子42、123、3407分别独立训练40,000 iter，并从各自 val 最优 checkpoint 复评。
- 每个种子的 val mIoU 必须不低于69.0，三种子均值必须不低于69.5。
- 三种子样本标准差必须不高于0.50 mIoU；否则认为训练稳定性不足，停止后续压缩研究。
- `summary.json` 必须明确记录 `selection_split=val` 和 `test_evaluated=false`；任一条件失败均不得进入 Phase 23。

### 实验设置

- 数据：CloudSEN12 High L1C，train 8,490张、val 535张；test 975张未评估。
- 输入与精度：512×512，FP32 Mask2Former训练，batch size 2；DINOv2-S冻结，Cloud-Adapter和压缩解码器可训练。
- 优化：AdamW，初始学习率1e-4、weight decay 0.05；500 iter线性 warmup，随后 PolyLR；梯度范数上限1.0。
- 训练：40,000 iter，每2,000 iter在 val评估并按 mIoU保存最佳 checkpoint。
- 硬件与环境：NVIDIA GeForce RTX 4090 D，PyTorch 2.1，conda环境 `cloud-lite-pt210`。
- 复现入口：`bash tools/run_phase22_oneclick_4090d.sh`；完整终端报告写入 `work_dirs/phase22_clean_v8/PHASE22_REPORT.txt`，机器可读汇总写入 `work_dirs/phase22_clean_v8/summary.json`。

PyTorch 2.1没有为 Mask2Former位置编码的 CUDA `cumsum`、deformable attention反传和 CUDA histogram提供确定性实现。本 Phase 固定 Python/NumPy/PyTorch种子，启用 cuDNN deterministic、关闭 benchmark，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；对上述无确定性实现的算子显式使用 `torch.use_deterministic_algorithms(True, warn_only=True)`。该例外同时应用于训练和验证并写入报告，不能把本结果描述为逐 bit 可复现。

### 实验结果

| Seed | Best checkpoint | aAcc | mIoU | mAcc | mDice | mPrecision | mRecall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `best_mIoU_iter_40000.pth` | 89.78 | 73.98 | 84.89 | 84.41 | 84.07 | 84.89 |
| 123 | `best_mIoU_iter_40000.pth` | 89.58 | 73.65 | 84.54 | 84.17 | 83.89 | 84.54 |
| 3407 | `best_mIoU_iter_40000.pth` | 89.66 | 73.78 | 84.63 | 84.27 | 83.98 | 84.63 |

| Seed | Clear IoU | Thick-cloud IoU | Thin-cloud IoU | Cloud-shadow IoU |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 88.56 | 85.09 | 60.80 | 61.47 |
| 123 | 88.32 | 84.91 | 58.77 | 62.59 |
| 3407 | 88.40 | 84.83 | 59.66 | 62.21 |

三种子 mIoU 为73.98、73.65、73.78，均值73.8033，样本标准差0.1662。三个硬门槛全部通过：最低单次结果比69.0高4.65点，均值比69.5高4.3033点，标准差比0.50低0.3338点。测试集没有被读取或评估。

### 运行修复记录

首次启动在严格确定性检查处暴露了两项环境兼容问题：cuBLAS需要显式 workspace配置，Mask2Former所用 CUDA算子在 PyTorch 2.1中没有确定性实现。修复分别固定 `CUBLAS_WORKSPACE_CONFIG`，并将预注册的 warn-only例外同时接入训练与验证。另增加 `VAL_EVAL_COMPLETE` 成功标记，避免 `tee` 在验证失败时留下的空壳日志被误判为完成。修复后复用已有 checkpoint继续执行，没有删除或伪造任何训练结果。

### 结论与下一步

Phase 22 通过。V8在清洁 val上的73.8033±0.1662 mIoU成为后续唯一基线；历史 Phase 1–20 的 test数值不得用于 Phase 23之后的方向选择。下一阶段可以进入预注册的结构创新筛选，但仍只能读取 val，并须相对本三种子基线报告同协议精度、复杂度和稳定性差异。

## Phase 23 — 标准算子 V12 配对基线

### Trial A：固定5倍 FPN 学习率

V12保留 Phase 22 V8的DINOv2-S、Cloud-Adapter、25个 query和两层 query decoder，仅将两层 deformable pixel encoder替换为标准算子的深度可分离 FPN。每个种子从同种子的清洁 V8 checkpoint warm-start；新FPN随机初始化并使用基础学习率的5倍，其余已训练模块使用2e-5。训练20,000 iter，每1,000 iter只在官方 val选取最佳 checkpoint，test保持封存。

预注册门槛为：每次 mIoU不低于66.5，三种子均值不低于67.0，样本标准差不高于0.60，且相对同种子 V8的平均 mIoU下降不超过3.0点。

| Seed | Best iter | V8 mIoU | V12 mIoU | Paired drop | Clear | Thick | Thin | Shadow |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 20,000 | 73.98 | 70.41 | 3.57 | 86.26 | 83.81 | 53.83 | 57.74 |
| 123 | 20,000 | 73.65 | 70.58 | 3.07 | 86.87 | 83.57 | 54.22 | 57.66 |
| 3407 | 16,000 | 73.78 | 70.67 | 3.11 | 86.57 | 83.63 | 55.10 | 57.38 |

V12平均 mIoU为70.5533、样本标准差0.1320，绝对精度和稳定性门槛均通过；但平均配对下降为3.2500，超过3.0止损线0.25点，因此 Trial A失败，不能进入 Phase 24。

### Repair A：唯一一次预注册学习率修复

按 Phase 23既定止损规则，不放宽3.0门槛，只允许一次短学习率筛选。使用 seed 42、相同 V8起点和官方 val，对FPN学习率倍数1、2、3、4分别训练8,000 iter。Trial A的5倍设置在前8,000 iter内最佳为67.58 mIoU；候选必须达到至少67.83（预先固定的+0.25改善）才有资格进入完整复现，合格者取val最高项，完全并列时取更低倍数。

若没有候选达到67.83，本 Phase立即关闭标准FPN方向；若存在候选，则冻结所选倍数，以原20,000 iter设置重跑42、123、3407三个配对种子，并原样复用Trial A全部门槛。Repair A再次失败后不得继续搜索学习率、训练长度、随机种子或阈值，后续转向保留V8 deformable decoder的结构化主干压缩。测试集在筛选和复现期间继续封存。

Repair A 的 seed-42 短筛结果如下：

| FPN LR multiplier | Best iter | Val mIoU | 相对5×同预算最佳67.58 | Qualified |
| ---: | ---: | ---: | ---: | --- |
| 1× | 3,000 | 49.64 | -17.94 | 否 |
| 2× | 8,000 | 62.46 | -5.12 | 否 |
| 3× | 8,000 | 64.73 | -2.85 | 否 |
| 4× | 8,000 | 65.61 | -1.97 | 否 |

四个候选均没有达到67.83资格线，且结果随倍数降低单调恶化；因此没有选择候选，也没有启动三种子确认。该结果反驳了“固定5倍FPN学习率过大导致Trial A差距”的修复假设。

### 结论与方向切换

Phase 23失败并正式关闭标准FPN替换分支。Trial A只比平均配对下降门槛多0.25点，但预注册的唯一修复没有任何候选接近资格线，不能事后放宽阈值或增加训练预算。V8 deformable decoder继续作为精度锚点；Phase 24不执行原先依赖V12通过的块裁剪方案，改为在保留V8解码器的前提下筛选结构化DINOv2主干块跳过。Phase 23全过程只使用官方val，test仍未评估。

## Phase 24 — V8 主干静态块裁剪零样本筛选

### 简介与门槛

Phase 23关闭标准FPN分支后，本 Phase回到通过清洁协议的 V8 seed-42 checkpoint，完整保留Cloud-Adapter和deformable Mask2Former，只把部分DINOv2 Transformer block静态替换为无参数 identity。所有适配器/输出位置 `[2,5,8,11]` 必须保留；筛选不训练，只读取官方val并用4090D原生PyTorch FP16、batch 1、512×512、warmup 20、正式100 iter测量速度。

基线必须在0.05 mIoU内复现Phase 22 seed 42；候选只有同时满足相对基线原生FP16平均加速不低于1.15倍、零样本val mIoU下降不超过8.0点才可进入后续微调。该8点仅为筛选容忍度，不是最终精度门槛。

### 实验结果

| Candidate | Active blocks | Params (M) | Val mIoU | Drop | Mean latency (ms) | Speedup | Peak GPU (GiB) | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline12 | 12 | 23.610 | 73.98 | 0.00 | 33.383 | 1.000× | 0.284 | 基线 |
| blocks10 | 10 | 20.060 | 56.17 | 17.81 | 31.878 | 1.047× | 0.278 | 否 |
| blocks8 | 8 | 16.510 | 40.41 | 33.57 | 30.293 | 1.102× | 0.271 | 否 |
| blocks6 | 6 | 12.959 | 34.94 | 39.04 | 28.557 | 1.169× | 0.263 | 否 |
| blocks4 | 4 | 9.409 | 13.47 | 60.51 | 27.044 | 1.234× | 0.276 | 否 |

12-block基线精确复现73.98 mIoU，所有延迟均为有限正值。10/8-block候选未达到1.15倍速度门槛且精度下降已经超过8点；6/4-block达到速度门槛，但分别损失39.04和60.51 mIoU。筛选没有合格候选，test未评估。

表中的checkpoint文件仍是包含原12-block权重的源 checkpoint（100.87 MiB），不能作为裁剪后产物大小；`parameters_m`才是运行图中实际保留的参数量。由于没有候选进入微调，本 Phase不生成或宣称紧凑部署checkpoint。

### 结论与方向切换

Phase 24失败并关闭“直接静态跳过预训练DINO块”方向。即使仅跳过两个非交互块，零样本精度也下降17.81点，而原生FP16只加速1.047倍；说明当前Cloud-Adapter交互依赖连续主干表征，且V8解码器/适配器开销限制了块裁剪的端到端收益。下一 Phase不微调这些候选，不放宽8点或1.15倍门槛，改为宽度/通道蒸馏或具备独立预训练权重的紧凑主干。

## Phase 25 — DINO MLP结构化通道裁剪筛选

### 简介与门槛

Phase 24证明直接跳过Transformer block既破坏精度又难以获得足够端到端加速。本 Phase保留全部12个注意力/残差块、Cloud-Adapter和V8 deformable Mask2Former，仅按路径强度
`||W_fc1[row]||2 × ||W_fc2[column]||2`静态保留每个DINO MLP的高分隐藏通道。候选MLP ratio为4.0控制组、3.0、2.5和2.0；不训练，只在官方val零样本筛选。候选必须同时达到相对控制组原生FP16平均加速至少1.08倍且mIoU下降不超过5.0点；控制组必须在0.05点内复现Phase 22 seed 42。测试集保持封存。

### 实验结果

| MLP ratio | Params (M) | Checkpoint (MiB) | Val mIoU | Drop | Mean latency (ms) | Speedup | Peak GPU (GiB) | Qualified |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4.0 | 23.610 | 90.204 | 73.98 | 0.00 | 34.261 | 1.000× | 0.304 | 控制组 |
| 3.0 | 20.067 | 76.686 | 43.80 | 30.18 | 34.226 | 1.001× | 0.294 | 否 |
| 2.5 | 18.295 | 69.927 | 34.74 | 39.24 | 34.541 | 0.992× | 0.271 | 否 |
| 2.0 | 16.523 | 63.169 | 30.42 | 43.56 | 34.231 | 1.001× | 0.268 | 否 |

4.0控制组精确复现73.98 mIoU，且全部延迟为有限正值。结构化裁剪确实把模型缩小到16.523M参数和63.169 MiB，但所有候选在4090D、batch 1、512×512、原生PyTorch FP16、20次warmup和100次正式计时下都没有获得可测的端到端加速；与此同时最轻的3.0裁剪已经损失30.18 mIoU。ratio 3.0/2.5/2.0的thin-cloud IoU分别降至20.74/1.01/0.00，说明无再训练的MLP通道选择严重破坏弱类表征。

### 结论与方向切换

Phase 25失败并关闭“从已训练DINOv2-S中直接裁剪MLP宽度”的方向。失败同时来自精度和真实延迟，不能通过放宽5点或1.08倍门槛修复；也不值得为这些零样本退化30点以上的候选投入三种子蒸馏。Phase 26按预注册止损切换到具有独立ImageNet预训练的紧凑主干，从结构起点获得真实卷积层级和硬件效率，而不是继续破坏DINO内部表示。全过程只读取官方val，test未评估。

## Phase 26 — ImageNet预训练ResNet-18紧凑主干试验

### 简介与门槛

Phase 26停止对DINOv2-S做破坏性后裁剪，改用具有独立ImageNet预训练的标准ResNetV1c-18层级主干，完整保留V8的128维、25-query、两层pixel encoder和两层query decoder。主干以解码器0.1倍学习率全量微调，seed 42训练40,000 iter；门槛为官方val mIoU至少68.0、同次复测V8原生FP16加速至少1.25倍、参数量不超过18.0M，且所有延迟有限为正。测试集保持封存。

### 实验结果

ResNet-18 seed-42最佳checkpoint在官方val达到68.02 mIoU、87.02 aAcc、78.96 mAcc和79.82 mDice；四类IoU依次为clear 85.38、thick cloud 82.68、thin cloud 47.72、cloud shadow 56.29。

| Model | Params (M) | Checkpoint (MiB) | Mean latency (ms) | Median (ms) | P90 (ms) | Throughput (img/s) | Peak GPU (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 22 V8 seed 42 | 23.610 | 100.874 | 34.264 | 34.114 | 34.722 | 29.185 | 0.304 |
| Phase 26 ResNet-18 seed 42 | 12.439 | 55.555 | 18.743 | 18.544 | 19.164 | 53.354 | 0.191 |

候选相对同次复测V8获得1.828倍平均延迟加速，参数减少47.32%，checkpoint减少44.93%，峰值显存减少37.14%。全部预注册门槛通过；精度门槛仅高0.02点，说明后续必须以三种子确认而不能把单种子边界通过直接当作稳定结论。

### 结论与下一步

Phase 26通过。结果支持“原生紧凑预训练结构比DINO事后裁剪更能形成真实精度—速度折中”，但68.02仍比Phase 22 seed-42低5.96点。Phase 27冻结网络、优化器、训练长度和阈值，只补充seed 123与3407，并复用本次seed 42组成三种子稳定性确认；不对ResNet-18追加事后调参。test仍未评估。

## Phase 27 — ResNet-18三种子稳定性确认

### 简介与门槛

Phase 27冻结Phase 26的结构、ImageNet初始化、优化器、学习率倍率和40,000 iter训练设置，复用seed 42并独立补充seed 123与3407。门槛为每次val mIoU不低于67.0、三种子均值不低于67.8、样本标准差不高于0.60，同时保持Phase 26冻结图的至少1.25倍加速和不超过18.0M参数。内部test继续封存。

### 实验结果

| Seed | Best iter | aAcc | mIoU | mAcc | mDice | Clear IoU | Thick IoU | Thin IoU | Shadow IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 30,000 | 87.02 | 68.02 | 78.96 | 79.82 | 85.38 | 82.68 | 47.72 | 56.29 |
| 123 | 34,000 | 86.99 | 68.05 | 79.07 | 79.79 | 85.19 | 83.09 | 46.39 | 57.53 |
| 3407 | 30,000 | 87.18 | 67.94 | 78.36 | 79.69 | 85.66 | 82.98 | 46.67 | 56.46 |

三种子mIoU均值为68.0033，样本标准差仅0.0569；最低一次67.94超过67.0门槛0.94点，均值超过67.8门槛0.2033点。冻结部署图仍为12.439M参数和1.828倍原生FP16加速，全部门槛通过。

### 结论与下一步

Phase 27通过。ResNet-18的精度明显低于Phase 22 V8的73.8033，但稳定性更高且获得可观真实加速，因此形成了可复现的速度—精度工作点。Phase 28在不训练、不调阈值的条件下，对三个V8和三个ResNet-18 checkpoint进行Landsat-8 Biome跨传感器零样本评估；预先固定CloudSEN类别到L8类别的映射，内部CloudSEN test仍不读取。

## Phase 28 — Landsat-8 Biome零样本跨传感器评估

### 简介与门槛

Phase 28不训练或校准模型，对Phase 22的三个V8与Phase 27的三个ResNet-18 checkpoint在完整Landsat-8 Biome test上进行零样本评估。类别映射在运行前固定为CloudSEN `(clear, thick, thin, shadow)` 到L8 `(clear, shadow, thin, cloud)` 的 `[0,3,2,1]`，不使用置信度阈值或测试时增强。门槛为每个紧凑模型至少35.0 mIoU、紧凑均值相对V8最多下降6.5点、三种子标准差不超过1.0。CloudSEN内部test继续封存。

### 实验结果

| Model | Seed 42 | Seed 123 | Seed 3407 | Mean mIoU | Sample std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 22 V8 | 39.33 | 40.03 | 39.90 | 39.7533 | 0.3723 |
| Phase 27 ResNet-18 | 33.10 | 31.89 | 33.34 | 32.7767 | 0.7772 |

ResNet-18三次结果均低于35.0绝对门槛；其外部均值比V8低6.9767点，超过6.5止损线0.4767点。标准差和有限数值检查通过，但四项实质门槛中两项失败。六次评估均覆盖2,643张L8测试图像，未读取CloudSEN内部test。

### 结论与方向切换

Phase 28失败并关闭ResNet-18主线。虽然它在CloudSEN val上稳定且快1.828倍，但跨传感器精度没有达到预注册下限，不能通过事后放宽阈值、改变类别映射或使用L8校准修复。L8 Biome在Phase 1–20历史工作区中已经存在，因此本结果应描述为外部跨传感器基准，而非从未接触的纯净外部holdout。Phase 29按止损规则转向独立ImageNet预训练的MobileNetV2原生移动主干，先做单种子内部精度、真实速度与L8零样本联合可行性试验。

## Phase 29 — MobileNetV2联合可行性试验

### 简介与门槛

Phase 29使用官方OpenMMLab ImageNet预训练MobileNetV2，确定性移除checkpoint的`backbone.`前缀和分类头后加载全部分割主干所需权重。保留V8解码器和Phase 26训练设置，seed 42训练40,000 iter。联合门槛为CloudSEN val mIoU至少66.0、L8零样本mIoU至少35.0、相对Phase 28 V8外部均值下降不超过6.5点、4090D原生FP16加速至少2.0倍、参数不超过7.5M。内部CloudSEN test继续封存。

### 实验结果

候选在CloudSEN官方val达到68.47 mIoU；四类IoU为clear 85.77、thick cloud 83.11、thin cloud 47.53、cloud shadow 57.46。L8 Biome零样本mIoU为33.86，比V8三种子外部均值39.7533低5.8933点。

| Model | Params (M) | Checkpoint (MiB) | Mean latency (ms) | Throughput (img/s) | Peak GPU (GiB) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 22 V8 | 23.610 | 100.874 | 34.264 | 29.185 | 0.304 | 1.000× |
| Phase 29 MobileNetV2 | 2.992 | 19.126 | 21.289 | 46.972 | 0.176 | 1.609× |

内部val精度、相对V8外部下降、参数量和有限数值门槛通过；但L8绝对mIoU低于35.0门槛1.14点，真实加速也低于2.0倍门槛0.391倍，因此联合试验失败。

### 结论与下一步

Phase 29失败并关闭本轮移动主干替换方向。MobileNetV2仅2.992M参数且内部val略高于ResNet-18，但V8 deformable解码器使端到端延迟不能随主干参数等比例下降，跨传感器绝对精度也未过线。按预注册不增加种子、不搜索宽度或学习率。Phase 30不再训练候选，而是冻结并审计Phase 21–29证据链，内部test不为失败候选解封。

## Phase 30 — 最终冻结证据审计

### 审计范围

Phase 30不训练、不推理、不读取CloudSEN内部test，只审计Phase 21–29摘要、统计量、checkpoint引用、测速口径、Git状态和复现测试。审计固定于commit `32538b31f2aaf2cc845f09db90d498128e6eb025`，并记录Phase 21与Phase 22–29全部机器摘要的SHA-256。

### 审计结果

41项仓库测试全部通过，1项按环境条件跳过。八项证据完整性门槛全部通过：Phase 21数据审计通过；Phase 22–29实际pass/fail与预期决策链一致；Phase 22与Phase 27三种子均值和样本标准差可从逐次记录精确复算；所引用checkpoint全部存在；Phase 21后内部test始终封存；效率比较均为batch 1、512×512、原生PyTorch FP16且驻留FP16权重；远程tracked worktree干净。

### 最终冻结结论

- 清洁精度锚点仍是Phase 22 V8：73.8033±0.1662 val mIoU。
- ResNet-18是可复现的内部效率点：68.0033±0.0569 val mIoU、1.828倍加速、12.439M参数；但它被Phase 28外部门槛拒绝，不能宣称为外部泛化合格模型。
- MobileNetV2 seed 42达到68.47内部val mIoU和2.992M参数，但L8仅33.86 mIoU且仅1.609倍加速，被Phase 29拒绝。
- 当前没有压缩后继同时通过清洁val、外部泛化和真实延迟联合门槛，因此没有候选有资格解封CloudSEN内部test。
- Landsat-8 Biome在历史工作区中已有，不能包装为从未观察的纯净外部holdout。TGRS投稿前仍需新增真正未用于开发的地域/外部数据，并提出能通过该证据链的方法。

Phase 30审计本身通过仅表示证据一致，不改变任何模型失败结论。完整单文件报告位于`work_dirs/phase30_final_audit/PHASE30_REPORT.txt`。

## Phase 31 — MobileNetV2 + LiteFPN标准算子解码器试验

### 简介与门槛

Phase 29表明MobileNetV2已经把模型压到2.992M参数，但两层deformable pixel decoder使端到端加速只有1.609倍。Phase 31保留128维、25-query、两层query decoder，只把pixel decoder替换为已有的深度可分离卷积`LiteFPNPixelDecoder`；使用同一ImageNet初始化、seed 42和40,000 iter重新训练，不迁移Phase 29分割权重。预注册门槛为官方val mIoU至少65.5、相对冻结V8基线的4090D原生FP16加速至少2.0倍、参数不超过3.0M，且所有数值有限。内部test继续封存。

### 实验结果

候选在官方val达到65.83 mIoU、85.82 aAcc、77.13 mAcc和78.02 mDice；四类IoU依次为clear 83.96、thick cloud 81.04、thin cloud 42.19、cloud shadow 56.15。

| Model | Params (M) | Checkpoint (MiB) | Mean latency (ms) | Median (ms) | P90 (ms) | Throughput (img/s) | Peak GPU (GiB) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 22 V8 | 23.610 | 100.874 | 34.264 | 34.114 | 34.722 | 29.185 | 0.304 | 1.000× |
| Phase 31 MobileNetV2 + LiteFPN | 2.548 | 17.950 | 16.710 | 16.310 | 16.921 | 59.846 | 0.180 | 2.051× |

全部预注册门槛通过：mIoU高于止损线0.33点，加速高于门槛0.051倍，参数量低于上限0.452M。相对Phase 29，LiteFPN使参数再减少0.444M，并把平均延迟从21.289 ms降至16.710 ms，首次形成满足内部精度、模型规模和两倍真实加速的紧凑候选；代价是内部val mIoU从68.47下降2.64点。

### 结论与下一步

Phase 31通过，但精度和速度余量均较窄，不能据此解封内部test。Phase 32冻结checkpoint、类别映射和推理设置，只进行一次Landsat-8 Biome零样本评估；门槛沿用既有外部协议：L8 mIoU至少35.0，且相对Phase 28 V8三种子均值39.7533下降不超过6.5点。若失败则关闭该结构，不做L8驱动调参；若通过，再补两个CloudSEN训练种子确认稳定性。

## Phase 32 — LiteFPN候选跨传感器零样本门控

### 简介与门槛

Phase 32冻结Phase 31 seed-42最佳checkpoint，不训练、不校准、不搜索阈值、不做测试时增强，也不重新选择checkpoint。在完整2,643张Landsat-8 Biome test图像上沿用既有`[0, 3, 2, 1]`类别映射进行一次512×512零样本评估。预注册门槛为L8 mIoU至少35.0、相对Phase 28 V8三种子均值39.7533下降不超过6.5点、全部指标有限且确实覆盖2,643张图像。CloudSEN内部test继续封存。

### 实验结果

候选获得35.01 mIoU、65.01 aAcc、51.07 mAcc和47.23 mDice；相对V8外部均值下降4.7433点。绝对mIoU以0.01点的极窄余量通过，均值下降余量为1.7567点；运行日志确认2,643/2,643张图像均已评估，全部数值有限，内部test未读取。

### 结论与下一步

Phase 32全部门槛通过。与Phase 29的MobileNetV2 + deformable decoder相比，LiteFPN版本的L8 mIoU从33.86升至35.01，同时内部val从68.47降至65.83，表明解码器替换改善了当前外部域折中，但外部绝对门槛仅高0.01点，结论必须谨慎。Phase 33冻结结构、训练超参数与门槛，只补充seed 123和3407，与既有seed 42组成CloudSEN val三种子稳定性确认；不再读取L8，不解封内部test。

## Phase 33 — LiteFPN三种子清洁验证稳定性

### 简介与门槛

Phase 33复用Phase 31 seed 42，并以完全相同的结构、ImageNet初始化、优化器和40,000 iter设置独立训练seed 123与3407。门槛为每个seed的官方val mIoU至少64.5、三种子均值至少65.3、样本标准差不超过0.75，同时复用同一部署图已测得的至少2.0倍加速和不超过3.0M参数。L8在本阶段不再读取，CloudSEN内部test继续封存。

### 实验结果

| Seed | Best iter | aAcc | mIoU | mAcc | mDice | Clear IoU | Thick IoU | Thin IoU | Shadow IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 30,000 | 85.82 | 65.83 | 77.13 | 78.02 | 83.96 | 81.04 | 42.19 | 56.15 |
| 123 | 34,000 | 87.24 | 68.00 | 78.30 | 79.74 | 85.29 | 83.76 | 47.12 | 55.84 |
| 3407 | 30,000 | 86.55 | 66.31 | 76.12 | 78.22 | 84.04 | 83.86 | 41.60 | 55.73 |

三种子均值为66.7133 mIoU，但样本标准差为1.1398；每次最低值和均值门槛均通过，标准差超过0.75止损线上限0.3898。冻结图仍为2.548M参数和2.051倍原生FP16加速，数值均有限。

### 结论与方向切换

Phase 33失败。该结构是有效的高效率工作点，但训练稳定性不足，而且三种子均值仍比Phase 22 V8的73.8033低7.09点，不能作为TGRS最终主模型。按预注册不放宽方差门槛、不挑选seed、不追加相同配方训练。Phase 34转向仅训练期存在的V8教师蒸馏：以Phase 31 seed-42学生热启动，采用弱类和边界加权语义KL，直接检验能否在不改变2.55M部署图的前提下把val恢复到至少68.0；本阶段不读取L8或内部test。

## Phase 34 — 弱类与边界加权语义蒸馏

### 简介与门槛

Phase 34以Phase 31 seed-42最佳checkpoint热启动学生，并以Phase 22 V8 seed-42作为冻结在线教师。固定temperature 2、KL权重1.0、类别权重`[1.0, 1.0, 2.5, 2.0]`、边界倍率1.5和半径1，训练10,000 iter。论文级门槛提高为val mIoU至少68.0、相对Phase 31配对提升至少1.5点、thin cloud与cloud shadow平均IoU至少51.0，并保持至少2.0倍原生FP16加速和不超过3.0M参数。L8和内部test均未读取。

### 实验结果

蒸馏候选仅达到61.44 mIoU，比Phase 31 seed-42反而下降4.39点；thin cloud和cloud shadow IoU分别为39.60与47.45，弱类均值43.525，低于51.0门槛7.475点。部署图保持2.548M参数，实测16.045 ms、2.135倍加速，但三个精度门槛全部失败。

### 结论与方向切换

Phase 34失败并关闭该固定蒸馏配方。结果与历史Phase 10普通logit KD和Phase 13特征KD的失败一致：强教师的输出分布会约束容量有限且结构不同的学生，弱类/边界加权没有解决结构错配，反而损害原有表征。按预注册不搜索温度、权重或训练长度。Phase 35改为部署图本身的单一结构增强：在LiteFPN的top-down路径后加入仅作用于stride-8及更粗尺度的轻量bottom-up融合，使细节返回query特征；从ImageNet初始化独立训练，不使用KD，不读取L8或内部test。

## Phase 35 — 双向LiteFPN结构增强试验

### 简介与门槛

Phase 35保留MobileNetV2及128维、25-query、两层query decoder，在LiteFPN top-down路径后增加stride-8及更粗尺度的bottom-up细节回流，每级使用非负归一化双输入权重和一个深度可分离细化块。seed 42从同一ImageNet初始化独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test未读取。

### 实验结果

候选达到66.61 mIoU，较Phase 31同seed提高0.78点；thin cloud与cloud shadow IoU为42.06和56.82，弱类均值49.44。模型为2.602M参数、17.854 ms、56.01 img/s，相对V8加速1.919倍。参数和有限数值门槛通过，但总体精度低1.39点、弱类低1.56点、速度低0.081倍，三项核心门槛全部失败。

### 结论与方向切换

Phase 35失败并关闭双向LiteFPN方向。bottom-up细化带来的0.78点精度增益不足以达到论文级68.0门槛，却消耗了Phase 31本就很窄的两倍加速余量，说明继续堆叠解码器融合不再划算。Phase 36停止修改解码器，回到通过独立ImageNet预训练结构提升表征效率：以torchvision MobileNetV3-Large四尺度主干替换MobileNetV2，配合原始单向LiteFPN，从头训练单种子联合检验精度、弱类和两倍加速。内部test继续封存。

## Phase 36 — MobileNetV3-Large + LiteFPN试验

### 简介与门槛

Phase 36使用torchvision MobileNetV3-Large IMAGENET1K_V2独立预训练权重，输出feature blocks 3/6/12/15的stride 4/8/16/32特征，配合Phase 31原始单向LiteFPN和128维、25-query、两层query decoder。seed 42独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过4.0M。L8与内部test未读取。

### 实验结果

候选达到65.96 mIoU；thin cloud和cloud shadow IoU为43.00和54.55，弱类均值48.775。模型为3.691M参数、17.491 ms、57.17 img/s，相对V8加速1.959倍。参数量和有限数值通过，但总体精度低2.04点、弱类低2.225点、速度低0.041倍。

### 结论与方向切换

Phase 36失败并关闭MobileNetV3-Large主干方向。它比Phase 31 MobileNetV2同seed高0.13点，却增加1.143M参数并失去两倍加速门槛，说明继续替换相近移动主干难以解决弱类表征瓶颈。Phase 37回到当前最快合格部署图Phase 31，不增加推理算子，改为直接监督的弱类加权query分类损失；与失败的教师KL不同，该实验只重新平衡真实标签监督，并从ImageNet初始化独立训练。若不能达到68.0则关闭损失重加权方向。

## Phase 37 — 弱类加权query分类监督

### 简介与门槛

Phase 37完全复用Phase 31部署结构、ImageNet初始化、优化器和40,000 iter训练，仅将Mask2Former query分类权重固定为clear/thick/thin/shadow/no-object=`[1.0, 1.0, 1.5, 1.25, 0.1]`。门槛为val mIoU至少68.0、thin cloud与cloud shadow平均IoU至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test未读取。

### 实验结果

候选达到67.35 mIoU，比Phase 31同seed提高1.52点；thin cloud与cloud shadow IoU为47.52和55.59，弱类均值51.555，首次超过51.0严格门槛。部署图仍为2.548M参数，实测16.997 ms、58.83 img/s和2.016倍加速。弱类、速度、参数与有限数值通过，但总体mIoU低于68.0门槛0.65点，因此按预注册总体判定失败。

### 结论与方向切换

Phase 37失败，不能把接近门槛包装为通过，也不调整已有类别权重。不过结果明确表明真实标签的弱类监督有效，而教师蒸馏无效。Phase 38关闭query分类权重路线并恢复原始统一query权重，转为训练期额外的像素语义交叉熵：对真实标签边界及弱类加权，直接约束最终semantic logits；该辅助分支不写入checkpoint、不改变Phase 31推理图。若仍不能达到68.0则停止监督损失变体。

## Phase 38 — 训练期边界语义辅助监督

### 简介与门槛

Phase 38恢复Phase 31的统一query分类权重，并在训练期对最终semantic logits增加像素交叉熵；类别权重固定为`[1.0, 1.0, 1.5, 1.25]`，边界倍率1.5、半径1、辅助损失权重0.5。该目标不引入checkpoint参数或推理算子，seed 42从同一ImageNet初始化独立训练40,000 iter。预注册门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test均未读取。

### 实验结果

候选达到66.78 mIoU；clear、thick cloud、thin cloud和cloud shadow IoU依次为85.58、80.83、45.93和54.77，弱类均值50.35。部署图为2.548M参数，4090D batch-1、512×512、原生FP16平均延迟16.179 ms、吞吐61.81 img/s，相对冻结V8基线34.264 ms为2.118倍加速。效率、参数、有限数值与内部test封存门槛通过，但总体mIoU低1.22点、弱类均值低0.65点。

### 结论与方向切换

Phase 38失败。它比Phase 31同seed提高0.95点，但比Phase 37低0.57点，且没有达到任何预注册精度门槛。按止损约定不改权重、边界半径、训练时长、seed或阈值，并关闭监督损失变体。Phase 39不再训练模型，而是冻结审计Phase 31–38的Pareto证据，以统一的论文级门槛判断是否存在可进入后续外部/内部测试的候选；若无候选，下一步必须先做冻结checkpoint的分层错误审计，禁止继续相邻超参数试错。

## Phase 39 — Phase 31–38冻结Pareto与证据审计

### 简介与门槛

Phase 39不训练、不推理、不读取内部test，只读取Phase 31–38的八份机器摘要并统一复核。审计门槛为：摘要齐全、历史pass/fail可复现、Phase 33均值与样本标准差可精确复算、测速均为batch 1/512×512/原生FP16且权重为FP16、所有统一指标有限、内部test始终封存。模型资格统一要求val mIoU至少68.0、弱类均值至少51.0、加速至少2.0倍、参数不超过3.0M，且已有三种子标准差不超过0.75并通过既有L8外部门槛。

### 审计结果

六项证据完整性门槛全部通过，但没有候选通过统一论文级门槛。Phase 37是最接近的紧凑点：67.35 val mIoU、51.555弱类均值、2.016倍加速、2.548M参数，仅总体精度低0.65点；Phase 38为66.78/50.35/2.118倍/2.548M。Phase 31虽然L8外部门槛通过，但三种子标准差1.1398超过0.75，且统一精度、弱类门槛均失败。其余Phase 34–36至少失败两项统一核心门槛。

### 冻结结论与下一步

审计通过只代表证据一致，`qualified_candidate=false`，不能解封CloudSEN内部test。类别权重微调、辅助监督损失、MobileNet家族替换、LiteFPN融合和教师KD五条相邻方向全部关闭。下一阶段按预注册先对最强紧凑checkpoint做完整val分层错误审计，定量区分弱类失败来自边界分辨率、类别稀有度还是场景上下文，再据此只选择一个新的方法方向；在诊断完成前不启动新训练。

## Phase 40 — 最强紧凑模型的冻结val分层错误审计

### 简介与门槛

Phase 40冻结Phase 37 seed-42最佳checkpoint和FP16推理图，对全部535张官方val图像做一次诊断性推理；按一像素标签边界/内部区域以及真实弱类像素占比四分位分层，不训练、不选择checkpoint、不读取L8或内部test。完整性门槛为535对样本唯一覆盖、聚合mIoU与冻结67.35相差不超过0.05点、各分层存在有效像素且指标有限、内部test封存。预注册决策规则优先检查“内部弱类召回减边界弱类召回”是否至少8点，其次检查最高/最低弱类占比四分位mIoU差是否至少8点，否则才转向上下文建模。

### 审计结果

全部完整性门槛通过，重新计算得到67.3542 mIoU和51.5540弱类mIoU，与Phase 37摘要一致。边界区域弱类召回仅46.3450，内部为69.1051，相差22.7601点，远超8点门槛；边界弱类mIoU为28.0166，内部为53.7590。按真实弱类占比从低到高，四组弱类mIoU依次为13.4825、37.6750、42.8415、50.8503，Q4-Q1差37.3677点，表明稀有度同样严重，但按预注册优先级不能越过更强的边界证据。

### 冻结结论与下一步

Phase 40选择`high_resolution_boundary_preserving_architecture`。这不是放宽Phase 38失败的边界损失：下一阶段只改推理结构，在原始stride-4特征上增加极轻量高频细节残差，并保持Phase 37其余结构与固定监督不变，以直接检验边界分辨率因果假设。Phase 41继续使用68.0总体mIoU、51.0弱类均值、2.0倍原生FP16加速、3.0M参数止损线；失败即关闭该结构，不调细节宽度或门控。

## Phase 41 — stride-4高频细节残差

### 简介与门槛

Phase 41保持Phase 37的MobileNetV2、LiteFPN、query decoder和冻结类别权重不变，只从原始stride-4特征增加24→16点卷积、3×3局部均值高通、16→128点卷积和初值0.1的单标量门控；残差仅注入mask feature，不改变query多尺度特征。seed 42从同一ImageNet初始化独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8和内部test未读取。

### 实验结果

候选达到67.26 mIoU；thin cloud和cloud shadow IoU分别为46.25与57.18，弱类均值51.715。模型为2.551M参数，4090D batch-1、512×512、原生FP16平均延迟16.635 ms、吞吐60.11 img/s，相对V8加速2.060倍。弱类、速度、参数和有限数值门槛通过，但总体mIoU低于68.0止损线0.74点；相对Phase 37总体还低0.09点，弱类仅高0.16点。

### 结论与方向切换

Phase 41失败。高频细节支路保住了两倍速度，也轻微改善弱类，但没有把Phase 40的边界诊断转化为总体精度收益；按预注册关闭该支路，不修改宽度、卷积核、门控初值、训练长度或seed。Phase 40的次级证据是弱类占比最低/最高四分位相差37.3677点，因此下一阶段转向训练数据稀有度采样：保持Phase 37模型与损失完全不变，仅用训练标签定义确定性的弱类富集采样，不读取val来构造采样权重。

## Phase 42 — 训练期弱类占比平衡采样

### 简介与门槛

Phase 42保持Phase 37的模型、损失、初始化、优化器、增强和40,000 iter完全不变。仅根据8,490张训练标注中thin cloud与cloud shadow的有效像素占比排序，以路径打破并列，取最高25%（2,123张，入池最低弱类占比25.5325%）为富集池；每次有50%概率从全训练集均匀有放回采样、50%从富集池均匀有放回采样。采样清单SHA-256为`a7993e99a9b56ebde7901873268d3ba3f2f6c79b7f8a9d707b93a33a859af8c8`。门槛仍为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8和内部test未读取。

### 实验结果

候选仅达到65.47 mIoU；thin cloud与cloud shadow IoU为43.71和54.80，弱类均值49.255。部署图未改变，为2.548M参数、16.716 ms、59.82 img/s，相对V8加速2.050倍。速度、参数和有限数值门槛通过，但总体mIoU低2.53点、弱类均值低1.745点；相对Phase 37总体下降1.88点、弱类下降2.30点。

### 结论与方向切换

Phase 42失败。弱类富集并未解决低占比场景，反而破坏了原始数据分布下的总体与弱类泛化；按预注册不调整富集分位、混合概率、替换规则、训练长度或seed，并关闭采样方向。Phase 40预注册决策树剩余的唯一方向是场景上下文建模。Phase 43将保持Phase 37配方，仅以最深stride-32特征生成全局通道门控，同时作用于LiteFPN各尺度；若仍失败，则停止该紧凑模型的相邻方法试验并冻结无合格候选结论。

## Phase 43 — 全局场景上下文门控LiteFPN

### 简介与门槛

Phase 43保持Phase 37全部训练配方，只对原始stride-32的320维特征做全局平均池化与320→128的1×1投影，以`2×sigmoid`生成共享通道门控并乘到LiteFPN四个尺度。投影权重和偏置零初始化，因此初始门控严格为1。seed 42独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test未读取。

### 实验结果

候选达到67.24 mIoU；thin cloud和cloud shadow IoU为47.16和54.38，弱类均值50.77。模型为2.589M参数、16.977 ms、58.90 img/s，相对V8加速2.018倍。速度、参数和有限数值门槛通过，但总体mIoU低0.76点、弱类低0.23点；相对Phase 37总体低0.11点、弱类低0.785点。

### 结论与止损

Phase 43失败。全局上下文门控没有越过Phase 37，且两倍加速余量已缩至0.018倍。按预注册关闭该方向，不修改上下文来源、投影宽度、初始化、门控位置、训练长度或seed，并停止紧凑模型相邻试错。Phase 44只做Phase 31–43冻结证据审计，确认摘要、测速口径、决策链和内部test封存状态；不再训练，也不以接近68.0包装成功。

## Phase 44 — Phase 31–43最终冻结证据审计

### 审计范围与门槛

Phase 44不训练、不推理、不读取CloudSEN内部test，只审计Phase 31–43共13份机器摘要。门槛为摘要齐全、历史pass/fail逐项吻合、内部test全程封存、所有实际测速均为batch 1/512×512/原生PyTorch FP16且权重为FP16、Phase 33统计可精确复算、候选指标有限、远程tracked worktree干净，并记录全部摘要SHA-256和审计commit。

### 审计结果

七项证据完整性门槛全部通过，`qualified_candidate=false`。最佳紧凑val点仍为Phase 37：67.35 mIoU、51.555弱类均值、2.016倍加速、2.548M参数，但总体精度低于68.0门槛0.65点；Phase 41弱类略高至51.715，但总体仅67.26。Phase 31虽通过既有L8门槛，其三种子标准差1.1398仍失败。Phase 34–38及41–43的蒸馏、融合、移动主干、监督、细节、采样和上下文变体均未形成联合合格候选。

### 最终冻结结论

Phase 31–43紧凑研究程序关闭。没有模型有资格解封CloudSEN内部test，也不能把Phase 37包装成达到68.0的成功模型。进一步研究必须提出不同的核心架构假设或引入真正未用于开发的外部/地域数据；禁止继续围绕现有MobileNetV2+LiteFPN做相邻损失、采样、门控或局部融合试错。Phase 44审计通过只代表证据链一致，不改变所有模型门槛结论。

## Phase 45 — 跨域可行性与上限审计（进行中）

### 预注册（结果产生前）

Phase 45开启与Phase 31–43不同的新研究主线：面向资源受限部署的弱类感知多类别跨传感器云分割域适应，核心假设为“物理语义因子化 + 类条件适应”。在实现新模型前先审计跨域任务是否有效并测量可利用上限。Landsat-8 Biome在历史阶段已被读取，只作为开发目标域，不能描述为独立外部holdout；CloudSEN内部test继续封存。

Phase 45A合并L8 Biome历史patch级train/test目录，从文件名恢复原始scene ID，再按biome分层、scene互斥地生成target-train、target-val与锁定target-test清单。共享数据文件只读，不移动、不复制、不改标签。完整性门槛为图像标注一一对应、文件名全部可解析、无重复patch、替代划分scene互斥、三个划分均覆盖全部类别和biome、标签值属于0–3且边界/内部像元统计有限。旧train/test的scene重叠只作为历史协议诊断，不作为新划分失败条件。

Phase 45A通过后才进入45B：冻结评估Phase 22 V8与Phase 37紧凑模型，使用target-train标签训练监督Oracle，并运行一个不读取目标标签的简单teacher–student适应基线。主要止损量为同架构Oracle target-val mIoU减source-only target-val mIoU：小于5点则停止；5到不足8点时仅当薄云/云影平均Oracle差距至少10点才继续；至少8点才进入因素化方法。锁定target-test在方法开发期间不读取，二分类HRC→GF实验只作独立补充，不与四分类Landsat指标直接求平均。

Phase 45B的Boundary F1定义在首次target-val模型推理前冻结：分别提取四类的一像素语义边界，以一像素空间容差匹配预测与真值边界，先聚合全体像素计算各类F1，再对四类宏平均。不得在观察结果后扩大容差。

### Phase 45A结果

训练机在commit `6587ae8`、`cloud-lite-pt210`环境中全量读取10,574对512×512图像与标注，所有配对、命名、标签范围、二维形状、重复检查和新划分完整性门槛通过，manifest SHA-256为`527fd1ecc81e0e088729626d6d2d1b2dc23fdd2bf901a33537c91822c8dcd533`。历史train为7,931 patches、88 scenes，历史test为2,643 patches、88 scenes；两者scene交集为88，即历史划分完全不满足scene互斥，后续禁止沿用该划分作DA证据。

新的biome分层scene级划分为target-train 6,502 patches/56 scenes、target-val 1,905/16、锁定target-test 2,167/16，三者均覆盖8个biome和四类且scene交集为空。target-train像元占比为clear 43.33%、cloud shadow 1.20%、thin cloud 19.31%、thick cloud 36.16%；target-val对应48.21%、2.82%、15.11%、33.86%。云影的极端稀缺支持弱类感知动机，但尚不构成方法有效性证据。Phase 45A通过，按预注册进入45B；target-test保持锁定。

### Phase 45B Source-only结果

在commit `683bb96`上冻结Phase 22 V8 seed-42与Phase 37 seed-42 checkpoint，使用固定`[0,3,2,1]`映射评估全部1,905张scene-disjoint target-val。V8为43.1241 mIoU，四类IoU依次为clear 74.7081、cloud shadow 16.4714、thin cloud 14.1403、thick cloud 67.1766，弱类均值15.3059，1-pixel容差宏Boundary F1为13.4763，其中thin cloud Boundary F1仅2.8838。Phase 37紧凑版为37.1549 mIoU，四类IoU为65.1102、15.8252、15.9285、51.7556，弱类均值15.8769，宏Boundary F1为11.5214，其中thin cloud为4.9800。两套模型均完成全部样本且指标有限；target-test与CloudSEN internal test均未读取。Source-only基准通过完整性检查，但低弱类与边界指标不代表方法成功，下一步必须用同协议Oracle测量真实上限后才能决定是否继续。

Oracle首次启动在commit `1f2157a`的训练前预检阶段失败：通用`check_light_setup.py`只识别目录式dataset并强制读取`data_root`，而Phase 45使用只读CSV manifest。失败发生在iter 0之前，未生成checkpoint、未产生可报告模型结果、未读取target-test。修复仅令预检器验证manifest文件存在，不改变数据、模型、优化器、训练长度或任何预注册门槛，随后原配置重新运行。

### Phase 45B Oracle结果

在commit `385aed5`、seed 42下，两套target-only Oracle均从通用DINO/ImageNet初始化训练40,000 iter，只使用scene-disjoint target-train标签并在target-val选择checkpoint。V8最佳为iter 32,000，在统一逐像素复算中达到56.2204 mIoU，四类IoU为clear 83.1552、cloud shadow 24.4730、thin cloud 39.4668、thick cloud 77.7865，弱类均值31.9699，宏Boundary F1为31.1843。相对冻结V8 source-only分别提升13.0963 mIoU、16.6640弱类IoU和17.7080 Boundary F1。

紧凑Oracle最佳为iter 40,000，达到53.0211 mIoU，四类IoU为83.7106、17.2911、37.0711、74.0118，弱类均值27.1811，宏Boundary F1为42.6916；相对Phase 37 source-only分别提升15.8663、11.3042和31.1702点。两套评估均完整覆盖1,905张target-val，指标有限，target-test与CloudSEN internal test均未读取。主判据V8 Oracle gap 13.0963超过预注册8点继续线，Phase 45B上限审计通过并选择`proceed_to_factorized_method`；在新方法前先完成预注册的简单teacher–student DA基线，禁止把Oracle监督结果作为DA结果。

## 后续维护规则

从 Phase 16 开始，每个 Phase 完成后在本文件末尾追加以下内容：

1. 简介与研究动机。
2. 明确、可证伪的实验目标和通过门槛。
3. 相对上一候选的网络、训练或部署改动。
4. 数据集、输入尺寸、精度、batch、硬件、warmup/iterations 等关键设置。
5. 完整分割指标或数值一致性指标，以及参数量、产物大小、延迟、吞吐和显存中实际测得的部分。
6. 与直接基线的差值、是否通过、失败原因和下一步决策。

禁止用局部样本 parity 代替完整数据集 mIoU，禁止把 fake quant 结果表述为真实 INT8 加速，禁止混用不同精度或不同计时口径而不加说明。
