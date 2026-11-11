# Cloud-Adapter 轻量化与边缘部署实验记录

> 汇总日期：2026-10-06。本文件合并了 `src` 根目录的 9 份阶段实验记录，保留全部实验正文、运行事件、指标、止损门与后续解释修正；整理只增加索引并调整标题层级，不改写原始实验数据。分散日志已封存，后续统一维护本文件。
>
> 原始文件（包括整理前的主日志及未提交修改）封存在 [phase_logs_originals_2026-10-06.zip](log_archives/phase_logs_originals_2026-10-06.zip)，包内 `manifest.json` 提供各原文件的字节数与 SHA-256。训练原始输出、预注册文件和研究计划仍保留原处。

## 阶段索引与来源

历史记录中的“待运行”“进行中”等状态按当时原文保留；同一阶段应结合后续完成记录、止损记录与解释修正阅读。下表是导航，详细科学结论及限制以正文为准。

| Phase | 正文入口 | 原始来源 | 最新记录摘要 |
| --- | --- | --- | --- |
| 1 | [Phase 1 — V1-LightHead 轻量基线](#phase-1) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 2 | [Phase 2 — Control 上限诊断与 V2 加权融合](#phase-2) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 3 | [Phase 3 — V3 轻量自顶向下 FPN](#phase-3) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 4 | [Phase 4 — V4 多尺度拼接融合](#phase-4) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 5 | [Phase 5 — V5 结构化压缩 Mask2Former](#phase-5) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 6 | [Phase 6 — V6 查询数减半](#phase-6) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 7 | [Phase 7 — V7 查询解码器减层](#phase-7) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 8 | [Phase 8 — V8 像素编码器减层](#phase-8) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 9 | [Phase 9 — V9-Micro 通道减半](#phase-9) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 10 | [Phase 10 — Control 到 V8 的语义 Logit 蒸馏](#phase-10) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 11 | [Phase 11 — FP16 与 W8 量化敏感性筛选](#phase-11) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 12 | [Phase 12 — V12 标准算子 ExportFPN](#phase-12) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 13 | [Phase 13 — V8 到 V12 的像素特征蒸馏](#phase-13) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 14 | [Phase 14 — V12 静态 ONNX 导出](#phase-14) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 15 | [Phase 15 — ONNX Runtime CUDA 一致性与性能验收](#phase-15) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 16 | [Phase 16 — TensorRT 多精度引擎构建与验收](#phase-16) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 17 | [Phase 17 — 完整测试集部署精度复核](#phase-17) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 18 | [Phase 18 — 真实端到端部署口径基准](#phase-18) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 19 | [Phase 19 — 输入分辨率精度—速度 Pareto 筛选](#phase-19) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 20 | [Phase 20 — 无 PyTorch 的原生掩膜 ONNX](#phase-20) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 21 | [Phase 21 — 无泄漏评估协议与数据完整性审计](#phase-21) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 22 | [Phase 22 — 清洁协议下的三种子 V8 基线](#phase-22) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 23 | [Phase 23 — 标准算子 V12 配对基线](#phase-23) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 24 | [Phase 24 — V8 主干静态块裁剪零样本筛选](#phase-24) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 25 | [Phase 25 — DINO MLP结构化通道裁剪筛选](#phase-25) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 26 | [Phase 26 — ImageNet预训练ResNet-18紧凑主干试验](#phase-26) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 27 | [Phase 27 — ResNet-18三种子稳定性确认](#phase-27) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 28 | [Phase 28 — Landsat-8 Biome零样本跨传感器评估](#phase-28) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 29 | [Phase 29 — MobileNetV2联合可行性试验](#phase-29) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 30 | [Phase 30 — 最终冻结证据审计](#phase-30) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 31 | [Phase 31 — MobileNetV2 + LiteFPN标准算子解码器试验](#phase-31) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 32 | [Phase 32 — LiteFPN候选跨传感器零样本门控](#phase-32) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 33 | [Phase 33 — LiteFPN三种子清洁验证稳定性](#phase-33) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 34 | [Phase 34 — 弱类与边界加权语义蒸馏](#phase-34) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 35 | [Phase 35 — 双向LiteFPN结构增强试验](#phase-35) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 36 | [Phase 36 — MobileNetV3-Large + LiteFPN试验](#phase-36) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 37 | [Phase 37 — 弱类加权query分类监督](#phase-37) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 38 | [Phase 38 — 训练期边界语义辅助监督](#phase-38) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 39 | [Phase 39 — Phase 31–38冻结Pareto与证据审计](#phase-39) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 40 | [Phase 40 — 最强紧凑模型的冻结val分层错误审计](#phase-40) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 41 | [Phase 41 — stride-4高频细节残差](#phase-41) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 42 | [Phase 42 — 训练期弱类占比平衡采样](#phase-42) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 43 | [Phase 43 — 全局场景上下文门控LiteFPN](#phase-43) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 44 | [Phase 44 — Phase 31–43最终冻结证据审计](#phase-44) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 45 | [Phase 45 — 跨域可行性与上限审计（进行中）](#phase-45) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 46 | [Phase 46 — 物理因素化输出空间可行性（进行中）](#phase-46) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 47 | [Phase 47 — mask feature独立因素头（进行中）](#phase-47) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 48 | [Phase 48 — 双视图因素条件原型适应（进行中）](#phase-48) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 49 | [Phase 49 — 源锚约束的无标签因素adapter（进行中）](#phase-49) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 50 | [Phase 50 — 1%目标标注主动适应（预注册）](#phase-50) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 51 | [Phase 51 — 跨传感器可信预测可行性审计（预注册）](#phase-51) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 52 | [Phase 52 — 稀疏目标传感器 MsRE 可行性（预注册）](#phase-52) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 52A-fix | [Phase 52A-fix — USGS shadow可用性修正版（预注册）](#phase-52a-fix) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 53 | [Phase 53 — 云影二元辅助监督（预注册）](#phase-53) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 52B | [Phase 52B — 冻结薄云路径的云影关系残差（预注册）](#phase-52b) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 55 | [Phase 55 — thin/shadow非对称负迁移机制审计（预注册）](#phase-55) | `EXPERIMENT_LOG.md` | 详见原主日志正文。 |
| 56–59 | [Phase 56–59](#phase-56-59) | `LOG56-59.md` | Phase 56 readout 联合门失败；57–59 未启动、止损封存。 |
| 60 | [Phase 60](#phase-60) | `LOG60.md` | 60A/60B 未找到当前协议下的联合解；保留后续解释边界修正。 |
| 61 | [Phase 61](#phase-61) | `LOG61.md` | 协议、优化与标签粒度审计及双盲复核完成；结果仅适用于选定错配位置。 |
| 61D2 | [Phase 61D2](#phase-61d2) | `LOG61D2.md` | 第三方裁决完成，结论为标签与模型的混合证据；保留缺理由、snow/ice 失败等限定。 |
| 62 | [Phase 62](#phase-62) | `LOG62.md` | 部分标签基线未改善弱类，停止复杂标签转移矩阵路线。 |
| 63 | [Phase 63](#phase-63) | `LOG63.md` | UI pilot 六门中五门通过，但 A 可定位率不足；不得启动正式 confirmation 或风险模型。 |
| 64 | [Phase 64](#phase-64) | `LOG64.md` | RGB 单 seed 快速筛选通过：MsRE 两域平均增益 +6.07，source 下降 0.04；尚非稳健性结论。 |
| 64B | [Phase 64B](#phase-64b) | `LOG64B.md` | 六波段源模型 mIoU 58.73，相对 RGB 下降 19.1238；源性能门失败，目标域实验未运行。 |

说明：原阶段日志未包含独立的 Phase 54 执行记录，未由研究计划补造实验结果。Phase 52A-fix、52B、61D2、64B 等分支均保留。


> 本文件是 Phase 实验的唯一持续记录。自 Phase 16 起，每完成一个 Phase，继续在本文件末尾追加，不另建实验记录文件。

- 原主日志最后更新：2026-09-10
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

<a id="phase-1"></a>

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

<a id="phase-2"></a>

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

<a id="phase-3"></a>

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

<a id="phase-4"></a>

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

<a id="phase-5"></a>

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

<a id="phase-6"></a>

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

<a id="phase-7"></a>

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

<a id="phase-8"></a>

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

<a id="phase-9"></a>

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

<a id="phase-10"></a>

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

<a id="phase-11"></a>

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

<a id="phase-12"></a>

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

<a id="phase-13"></a>

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

<a id="phase-14"></a>

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

<a id="phase-15"></a>

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

<a id="phase-16"></a>

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

<a id="phase-17"></a>

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

<a id="phase-18"></a>

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

<a id="phase-19"></a>

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

<a id="phase-20"></a>

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

<a id="phase-21"></a>

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

<a id="phase-22"></a>

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

<a id="phase-23"></a>

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

<a id="phase-24"></a>

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

<a id="phase-25"></a>

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

<a id="phase-26"></a>

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

<a id="phase-27"></a>

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

<a id="phase-28"></a>

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

<a id="phase-29"></a>

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

<a id="phase-30"></a>

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

<a id="phase-31"></a>

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

<a id="phase-32"></a>

## Phase 32 — LiteFPN候选跨传感器零样本门控

### 简介与门槛

Phase 32冻结Phase 31 seed-42最佳checkpoint，不训练、不校准、不搜索阈值、不做测试时增强，也不重新选择checkpoint。在完整2,643张Landsat-8 Biome test图像上沿用既有`[0, 3, 2, 1]`类别映射进行一次512×512零样本评估。预注册门槛为L8 mIoU至少35.0、相对Phase 28 V8三种子均值39.7533下降不超过6.5点、全部指标有限且确实覆盖2,643张图像。CloudSEN内部test继续封存。

### 实验结果

候选获得35.01 mIoU、65.01 aAcc、51.07 mAcc和47.23 mDice；相对V8外部均值下降4.7433点。绝对mIoU以0.01点的极窄余量通过，均值下降余量为1.7567点；运行日志确认2,643/2,643张图像均已评估，全部数值有限，内部test未读取。

### 结论与下一步

Phase 32全部门槛通过。与Phase 29的MobileNetV2 + deformable decoder相比，LiteFPN版本的L8 mIoU从33.86升至35.01，同时内部val从68.47降至65.83，表明解码器替换改善了当前外部域折中，但外部绝对门槛仅高0.01点，结论必须谨慎。Phase 33冻结结构、训练超参数与门槛，只补充seed 123和3407，与既有seed 42组成CloudSEN val三种子稳定性确认；不再读取L8，不解封内部test。

<a id="phase-33"></a>

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

<a id="phase-34"></a>

## Phase 34 — 弱类与边界加权语义蒸馏

### 简介与门槛

Phase 34以Phase 31 seed-42最佳checkpoint热启动学生，并以Phase 22 V8 seed-42作为冻结在线教师。固定temperature 2、KL权重1.0、类别权重`[1.0, 1.0, 2.5, 2.0]`、边界倍率1.5和半径1，训练10,000 iter。论文级门槛提高为val mIoU至少68.0、相对Phase 31配对提升至少1.5点、thin cloud与cloud shadow平均IoU至少51.0，并保持至少2.0倍原生FP16加速和不超过3.0M参数。L8和内部test均未读取。

### 实验结果

蒸馏候选仅达到61.44 mIoU，比Phase 31 seed-42反而下降4.39点；thin cloud和cloud shadow IoU分别为39.60与47.45，弱类均值43.525，低于51.0门槛7.475点。部署图保持2.548M参数，实测16.045 ms、2.135倍加速，但三个精度门槛全部失败。

### 结论与方向切换

Phase 34失败并关闭该固定蒸馏配方。结果与历史Phase 10普通logit KD和Phase 13特征KD的失败一致：强教师的输出分布会约束容量有限且结构不同的学生，弱类/边界加权没有解决结构错配，反而损害原有表征。按预注册不搜索温度、权重或训练长度。Phase 35改为部署图本身的单一结构增强：在LiteFPN的top-down路径后加入仅作用于stride-8及更粗尺度的轻量bottom-up融合，使细节返回query特征；从ImageNet初始化独立训练，不使用KD，不读取L8或内部test。

<a id="phase-35"></a>

## Phase 35 — 双向LiteFPN结构增强试验

### 简介与门槛

Phase 35保留MobileNetV2及128维、25-query、两层query decoder，在LiteFPN top-down路径后增加stride-8及更粗尺度的bottom-up细节回流，每级使用非负归一化双输入权重和一个深度可分离细化块。seed 42从同一ImageNet初始化独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test未读取。

### 实验结果

候选达到66.61 mIoU，较Phase 31同seed提高0.78点；thin cloud与cloud shadow IoU为42.06和56.82，弱类均值49.44。模型为2.602M参数、17.854 ms、56.01 img/s，相对V8加速1.919倍。参数和有限数值门槛通过，但总体精度低1.39点、弱类低1.56点、速度低0.081倍，三项核心门槛全部失败。

### 结论与方向切换

Phase 35失败并关闭双向LiteFPN方向。bottom-up细化带来的0.78点精度增益不足以达到论文级68.0门槛，却消耗了Phase 31本就很窄的两倍加速余量，说明继续堆叠解码器融合不再划算。Phase 36停止修改解码器，回到通过独立ImageNet预训练结构提升表征效率：以torchvision MobileNetV3-Large四尺度主干替换MobileNetV2，配合原始单向LiteFPN，从头训练单种子联合检验精度、弱类和两倍加速。内部test继续封存。

<a id="phase-36"></a>

## Phase 36 — MobileNetV3-Large + LiteFPN试验

### 简介与门槛

Phase 36使用torchvision MobileNetV3-Large IMAGENET1K_V2独立预训练权重，输出feature blocks 3/6/12/15的stride 4/8/16/32特征，配合Phase 31原始单向LiteFPN和128维、25-query、两层query decoder。seed 42独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过4.0M。L8与内部test未读取。

### 实验结果

候选达到65.96 mIoU；thin cloud和cloud shadow IoU为43.00和54.55，弱类均值48.775。模型为3.691M参数、17.491 ms、57.17 img/s，相对V8加速1.959倍。参数量和有限数值通过，但总体精度低2.04点、弱类低2.225点、速度低0.041倍。

### 结论与方向切换

Phase 36失败并关闭MobileNetV3-Large主干方向。它比Phase 31 MobileNetV2同seed高0.13点，却增加1.143M参数并失去两倍加速门槛，说明继续替换相近移动主干难以解决弱类表征瓶颈。Phase 37回到当前最快合格部署图Phase 31，不增加推理算子，改为直接监督的弱类加权query分类损失；与失败的教师KL不同，该实验只重新平衡真实标签监督，并从ImageNet初始化独立训练。若不能达到68.0则关闭损失重加权方向。

<a id="phase-37"></a>

## Phase 37 — 弱类加权query分类监督

### 简介与门槛

Phase 37完全复用Phase 31部署结构、ImageNet初始化、优化器和40,000 iter训练，仅将Mask2Former query分类权重固定为clear/thick/thin/shadow/no-object=`[1.0, 1.0, 1.5, 1.25, 0.1]`。门槛为val mIoU至少68.0、thin cloud与cloud shadow平均IoU至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test未读取。

### 实验结果

候选达到67.35 mIoU，比Phase 31同seed提高1.52点；thin cloud与cloud shadow IoU为47.52和55.59，弱类均值51.555，首次超过51.0严格门槛。部署图仍为2.548M参数，实测16.997 ms、58.83 img/s和2.016倍加速。弱类、速度、参数与有限数值通过，但总体mIoU低于68.0门槛0.65点，因此按预注册总体判定失败。

### 结论与方向切换

Phase 37失败，不能把接近门槛包装为通过，也不调整已有类别权重。不过结果明确表明真实标签的弱类监督有效，而教师蒸馏无效。Phase 38关闭query分类权重路线并恢复原始统一query权重，转为训练期额外的像素语义交叉熵：对真实标签边界及弱类加权，直接约束最终semantic logits；该辅助分支不写入checkpoint、不改变Phase 31推理图。若仍不能达到68.0则停止监督损失变体。

<a id="phase-38"></a>

## Phase 38 — 训练期边界语义辅助监督

### 简介与门槛

Phase 38恢复Phase 31的统一query分类权重，并在训练期对最终semantic logits增加像素交叉熵；类别权重固定为`[1.0, 1.0, 1.5, 1.25]`，边界倍率1.5、半径1、辅助损失权重0.5。该目标不引入checkpoint参数或推理算子，seed 42从同一ImageNet初始化独立训练40,000 iter。预注册门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test均未读取。

### 实验结果

候选达到66.78 mIoU；clear、thick cloud、thin cloud和cloud shadow IoU依次为85.58、80.83、45.93和54.77，弱类均值50.35。部署图为2.548M参数，4090D batch-1、512×512、原生FP16平均延迟16.179 ms、吞吐61.81 img/s，相对冻结V8基线34.264 ms为2.118倍加速。效率、参数、有限数值与内部test封存门槛通过，但总体mIoU低1.22点、弱类均值低0.65点。

### 结论与方向切换

Phase 38失败。它比Phase 31同seed提高0.95点，但比Phase 37低0.57点，且没有达到任何预注册精度门槛。按止损约定不改权重、边界半径、训练时长、seed或阈值，并关闭监督损失变体。Phase 39不再训练模型，而是冻结审计Phase 31–38的Pareto证据，以统一的论文级门槛判断是否存在可进入后续外部/内部测试的候选；若无候选，下一步必须先做冻结checkpoint的分层错误审计，禁止继续相邻超参数试错。

<a id="phase-39"></a>

## Phase 39 — Phase 31–38冻结Pareto与证据审计

### 简介与门槛

Phase 39不训练、不推理、不读取内部test，只读取Phase 31–38的八份机器摘要并统一复核。审计门槛为：摘要齐全、历史pass/fail可复现、Phase 33均值与样本标准差可精确复算、测速均为batch 1/512×512/原生FP16且权重为FP16、所有统一指标有限、内部test始终封存。模型资格统一要求val mIoU至少68.0、弱类均值至少51.0、加速至少2.0倍、参数不超过3.0M，且已有三种子标准差不超过0.75并通过既有L8外部门槛。

### 审计结果

六项证据完整性门槛全部通过，但没有候选通过统一论文级门槛。Phase 37是最接近的紧凑点：67.35 val mIoU、51.555弱类均值、2.016倍加速、2.548M参数，仅总体精度低0.65点；Phase 38为66.78/50.35/2.118倍/2.548M。Phase 31虽然L8外部门槛通过，但三种子标准差1.1398超过0.75，且统一精度、弱类门槛均失败。其余Phase 34–36至少失败两项统一核心门槛。

### 冻结结论与下一步

审计通过只代表证据一致，`qualified_candidate=false`，不能解封CloudSEN内部test。类别权重微调、辅助监督损失、MobileNet家族替换、LiteFPN融合和教师KD五条相邻方向全部关闭。下一阶段按预注册先对最强紧凑checkpoint做完整val分层错误审计，定量区分弱类失败来自边界分辨率、类别稀有度还是场景上下文，再据此只选择一个新的方法方向；在诊断完成前不启动新训练。

<a id="phase-40"></a>

## Phase 40 — 最强紧凑模型的冻结val分层错误审计

### 简介与门槛

Phase 40冻结Phase 37 seed-42最佳checkpoint和FP16推理图，对全部535张官方val图像做一次诊断性推理；按一像素标签边界/内部区域以及真实弱类像素占比四分位分层，不训练、不选择checkpoint、不读取L8或内部test。完整性门槛为535对样本唯一覆盖、聚合mIoU与冻结67.35相差不超过0.05点、各分层存在有效像素且指标有限、内部test封存。预注册决策规则优先检查“内部弱类召回减边界弱类召回”是否至少8点，其次检查最高/最低弱类占比四分位mIoU差是否至少8点，否则才转向上下文建模。

### 审计结果

全部完整性门槛通过，重新计算得到67.3542 mIoU和51.5540弱类mIoU，与Phase 37摘要一致。边界区域弱类召回仅46.3450，内部为69.1051，相差22.7601点，远超8点门槛；边界弱类mIoU为28.0166，内部为53.7590。按真实弱类占比从低到高，四组弱类mIoU依次为13.4825、37.6750、42.8415、50.8503，Q4-Q1差37.3677点，表明稀有度同样严重，但按预注册优先级不能越过更强的边界证据。

### 冻结结论与下一步

Phase 40选择`high_resolution_boundary_preserving_architecture`。这不是放宽Phase 38失败的边界损失：下一阶段只改推理结构，在原始stride-4特征上增加极轻量高频细节残差，并保持Phase 37其余结构与固定监督不变，以直接检验边界分辨率因果假设。Phase 41继续使用68.0总体mIoU、51.0弱类均值、2.0倍原生FP16加速、3.0M参数止损线；失败即关闭该结构，不调细节宽度或门控。

<a id="phase-41"></a>

## Phase 41 — stride-4高频细节残差

### 简介与门槛

Phase 41保持Phase 37的MobileNetV2、LiteFPN、query decoder和冻结类别权重不变，只从原始stride-4特征增加24→16点卷积、3×3局部均值高通、16→128点卷积和初值0.1的单标量门控；残差仅注入mask feature，不改变query多尺度特征。seed 42从同一ImageNet初始化独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8和内部test未读取。

### 实验结果

候选达到67.26 mIoU；thin cloud和cloud shadow IoU分别为46.25与57.18，弱类均值51.715。模型为2.551M参数，4090D batch-1、512×512、原生FP16平均延迟16.635 ms、吞吐60.11 img/s，相对V8加速2.060倍。弱类、速度、参数和有限数值门槛通过，但总体mIoU低于68.0止损线0.74点；相对Phase 37总体还低0.09点，弱类仅高0.16点。

### 结论与方向切换

Phase 41失败。高频细节支路保住了两倍速度，也轻微改善弱类，但没有把Phase 40的边界诊断转化为总体精度收益；按预注册关闭该支路，不修改宽度、卷积核、门控初值、训练长度或seed。Phase 40的次级证据是弱类占比最低/最高四分位相差37.3677点，因此下一阶段转向训练数据稀有度采样：保持Phase 37模型与损失完全不变，仅用训练标签定义确定性的弱类富集采样，不读取val来构造采样权重。

<a id="phase-42"></a>

## Phase 42 — 训练期弱类占比平衡采样

### 简介与门槛

Phase 42保持Phase 37的模型、损失、初始化、优化器、增强和40,000 iter完全不变。仅根据8,490张训练标注中thin cloud与cloud shadow的有效像素占比排序，以路径打破并列，取最高25%（2,123张，入池最低弱类占比25.5325%）为富集池；每次有50%概率从全训练集均匀有放回采样、50%从富集池均匀有放回采样。采样清单SHA-256为`a7993e99a9b56ebde7901873268d3ba3f2f6c79b7f8a9d707b93a33a859af8c8`。门槛仍为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8和内部test未读取。

### 实验结果

候选仅达到65.47 mIoU；thin cloud与cloud shadow IoU为43.71和54.80，弱类均值49.255。部署图未改变，为2.548M参数、16.716 ms、59.82 img/s，相对V8加速2.050倍。速度、参数和有限数值门槛通过，但总体mIoU低2.53点、弱类均值低1.745点；相对Phase 37总体下降1.88点、弱类下降2.30点。

### 结论与方向切换

Phase 42失败。弱类富集并未解决低占比场景，反而破坏了原始数据分布下的总体与弱类泛化；按预注册不调整富集分位、混合概率、替换规则、训练长度或seed，并关闭采样方向。Phase 40预注册决策树剩余的唯一方向是场景上下文建模。Phase 43将保持Phase 37配方，仅以最深stride-32特征生成全局通道门控，同时作用于LiteFPN各尺度；若仍失败，则停止该紧凑模型的相邻方法试验并冻结无合格候选结论。

<a id="phase-43"></a>

## Phase 43 — 全局场景上下文门控LiteFPN

### 简介与门槛

Phase 43保持Phase 37全部训练配方，只对原始stride-32的320维特征做全局平均池化与320→128的1×1投影，以`2×sigmoid`生成共享通道门控并乘到LiteFPN四个尺度。投影权重和偏置零初始化，因此初始门控严格为1。seed 42独立训练40,000 iter。门槛为val mIoU至少68.0、弱类均值至少51.0、原生FP16加速至少2.0倍、参数不超过3.0M。L8与内部test未读取。

### 实验结果

候选达到67.24 mIoU；thin cloud和cloud shadow IoU为47.16和54.38，弱类均值50.77。模型为2.589M参数、16.977 ms、58.90 img/s，相对V8加速2.018倍。速度、参数和有限数值门槛通过，但总体mIoU低0.76点、弱类低0.23点；相对Phase 37总体低0.11点、弱类低0.785点。

### 结论与止损

Phase 43失败。全局上下文门控没有越过Phase 37，且两倍加速余量已缩至0.018倍。按预注册关闭该方向，不修改上下文来源、投影宽度、初始化、门控位置、训练长度或seed，并停止紧凑模型相邻试错。Phase 44只做Phase 31–43冻结证据审计，确认摘要、测速口径、决策链和内部test封存状态；不再训练，也不以接近68.0包装成功。

<a id="phase-44"></a>

## Phase 44 — Phase 31–43最终冻结证据审计

### 审计范围与门槛

Phase 44不训练、不推理、不读取CloudSEN内部test，只审计Phase 31–43共13份机器摘要。门槛为摘要齐全、历史pass/fail逐项吻合、内部test全程封存、所有实际测速均为batch 1/512×512/原生PyTorch FP16且权重为FP16、Phase 33统计可精确复算、候选指标有限、远程tracked worktree干净，并记录全部摘要SHA-256和审计commit。

### 审计结果

七项证据完整性门槛全部通过，`qualified_candidate=false`。最佳紧凑val点仍为Phase 37：67.35 mIoU、51.555弱类均值、2.016倍加速、2.548M参数，但总体精度低于68.0门槛0.65点；Phase 41弱类略高至51.715，但总体仅67.26。Phase 31虽通过既有L8门槛，其三种子标准差1.1398仍失败。Phase 34–38及41–43的蒸馏、融合、移动主干、监督、细节、采样和上下文变体均未形成联合合格候选。

### 最终冻结结论

Phase 31–43紧凑研究程序关闭。没有模型有资格解封CloudSEN内部test，也不能把Phase 37包装成达到68.0的成功模型。进一步研究必须提出不同的核心架构假设或引入真正未用于开发的外部/地域数据；禁止继续围绕现有MobileNetV2+LiteFPN做相邻损失、采样、门控或局部融合试错。Phase 44审计通过只代表证据链一致，不改变所有模型门槛结论。

<a id="phase-45"></a>

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

### Phase 45B Teacher–student预注册（结果产生前）

简单source-free基线冻结Phase 22 seed-42 V8作为离线教师，仅对6,502张target-train图像推理且不读取其标注。将教师四类分数softmax后以0.90置信度保留source-order伪标签，单图有效像元不足5%则排除；伪标签必须至少保留1,000张图且四类均有有效像元。学生从同一source checkpoint初始化，以随机翻转和颜色扰动训练固定10,000 iter，不运行target-val选择，固定评估iter-10,000。通过线为target-val相对V8 source-only至少提升1.0 mIoU，同时CloudSEN source-val遗忘不超过1.0 mIoU。失败即关闭固定教师伪标签方向并转入物理语义因子化/类条件适应；成功只保留为DA baseline，不构成论文主方法。

### Phase 45B Teacher–student结果与止损

在commit `ed75dc7`上，冻结教师完整推理6,502张target-train图像。0.90阈值仅接受1,385个像元，占全部目标训练像元`8.125739330806867e-07`，且全部来自source-order类别0；四类教师预测像元数为`[625157244, 814578776, 160874855, 103849413]`，对应被接受数为`[1385, 0, 0, 0]`。没有任何图像达到5%最小覆盖率，因此保留图像数为0，至少1,000张和四类均被接受两项门槛均失败。整个过程未读取target-train标签、target-val标签、锁定target-test或CloudSEN internal test；失败发生在学生训练前，没有生成学生checkpoint。

该结果说明Mask2Former聚合后的四类分数在跨传感器目标域上不能按固定0.90 softmax阈值充当可用置信度，而不是说明目标域没有可学习信号；同架构Oracle的13.0963 mIoU差距已独立证明后者存在。按预注册不事后降低阈值、不改最小覆盖率、不引入类别特定阈值，也不启动10,000 iter学生训练。固定教师硬伪标签方向正式关闭，Phase 45完成：跨域任务与上限成立，但朴素自训练基线失败。下一阶段只进入预先声明的“物理语义因子化 + 类条件适应”，首先用单一Phase验证因素监督与合法四类重构是否能在保持源域的同时改善目标域弱类，再决定是否加入目标原型更新。

<a id="phase-46"></a>

## Phase 46 — 物理因素化输出空间可行性（进行中）

### 预注册（结果产生前）

Phase 46只回答因素化表示能否作为后续适应的稳定底座，不使用目标训练标签，也不同时引入原型。冻结Phase 22 V8 seed-42全部原参数，在四类semantic logits之上增加一个零初始化的4→3点卷积残差校准器。三个因素依次为阴影/非阴影、非阴影内的有云/clear，以及有云条件下的厚云/薄云；通过条件概率乘法只重构`clear、thick、thin、shadow`四个合法状态。零残差时重构概率与原四类softmax在数值上严格等价，避免以随机新头破坏基线。

仅使用CloudSEN source-train标签训练4,000 iter，每1,000 iter在官方source-val选择checkpoint；因素BCE与合法四类重构NLL权重均固定为1.0，只有15个校准器参数可训练。训练后一次性评估source-val和scene-disjoint target-val；target-train、锁定target-test及CloudSEN internal test均不读取。通过门槛为source-val mIoU不低于72.98（相对同seed 73.98遗忘不超过1.0）、target-val mIoU不低于43.1241、target弱类mIoU至少16.3059（相对15.3059提升至少1.0），且重构概率和为1、预测只包含四个合法状态。全部通过才允许Phase 47加入类条件目标原型；任一失败则关闭“仅在最终四类logits上做因素残差校准”的实现，不追加训练时长、权重或学习率试验，转而把独立因素头前移到像素特征空间。

### 实验结果与止损

seed 42固定训练4,000 iter，source-val选择iter 4,000。正确经过segmentor因素重构的完整535张source-val为72.6871 mIoU，四类IoU依次为clear 88.0634、thick cloud 85.0886、thin cloud 57.1206、cloud shadow 60.4757，相对同seed V8下降1.2929点，超过1.0遗忘线0.2929点。完整1,905张scene-disjoint target-val为42.5357 mIoU，四类IoU依次为clear 76.9311、cloud shadow 16.7143、thin cloud 10.0850、thick cloud 66.4123，弱类均值13.3997，宏Boundary F1为13.3340；相对source-only分别下降0.5884 mIoU、1.9062弱类IoU和0.1423 Boundary F1。零残差数学重构的最大概率误差为`3.33e-16`、概率和最大误差为`2.22e-16`，结构合法性通过，但三项性能门槛全部失败。target-train标签、锁定target-test与CloudSEN internal test均未读取。

首次机器汇总错误复用了为普通Mask2Former编写的部署wrapper，绕过segmentor级因素重构，因而机械复现V8原值；训练期mmseg验证与该汇总不一致后即被识别。修复只令评估调用已有checkpoint的`encode_decode`，未重训、未改变checkpoint或门槛；以上为修复后的唯一有效结果。Phase 46失败并关闭最终四类logits上的15参数因素残差校准，不延长训练或调权重。结果表明因素监督必须作用于共享像素表征，而不能在已经压缩为四类分数后恢复跨传感器弱类信息；下一阶段按预注册将独立因素头前移到mask feature空间，仍先验证源域保持与目标零样本弱类，再决定是否允许目标原型更新。

<a id="phase-47"></a>

## Phase 47 — mask feature独立因素头（进行中）

### 预注册（结果产生前）

Phase 47保持Phase 46的数据、V8 checkpoint、冻结策略、因素定义、合法四类条件重构、优化器、4,000 iter和source-val选择规则不变；唯一方法变量是将3个因素残差从最终四类logits前移到Mask2Former stride-4、128维mask feature，以1×1卷积直接读取共享像素表征。该头共387个可训练参数并继续零初始化，所以iter 0与V8概率严格等价。它检验Phase 46失败是否确由四类压缩后的信息瓶颈造成，而不是同时试验目标伪标签或原型。

门槛完全沿用Phase 46：source-val mIoU至少72.98，scene-disjoint target-val mIoU至少43.1241，target弱类mIoU至少16.3059，并通过概率合法性、样本覆盖和封存检查。全部通过才允许下一阶段加入类条件目标原型；任一失败即关闭“纯source监督的零样本因素残差”方向，不调整特征层、卷积核、训练长度、学习率或损失权重，下一方向必须直接检验目标无标签信息能否在严格防塌缩约束下提供适应增益。

### 实验结果与止损

source-val在iter 1,000达到全程最佳73.2179 mIoU，四类IoU为clear 88.0599、thick cloud 84.6370、thin cloud 58.8692、cloud shadow 61.3057，相对同seed V8遗忘0.7621点，源域保持门槛通过。冻结该checkpoint评估1,905张target-val得到42.8517 mIoU，四类IoU为clear 75.8086、cloud shadow 16.6329、thin cloud 12.2356、thick cloud 66.7297，弱类均值14.4342、宏Boundary F1 13.2528。相对source-only分别下降0.2724 mIoU、0.8717弱类IoU和0.2235 Boundary F1；目标总体与弱类门槛均失败。结构合法性、样本覆盖、有限数值和全部封存检查通过，target-train标签、target-test与CloudSEN internal test均未读取。

Phase 47失败。将因素头前移到mask feature后源域明显优于Phase 46，说明共享像素表征确有可用因素信息；但没有目标数据约束时，该信息仍沿源域决策边界组织，不能自动缩小传感器差异。按预注册关闭纯source监督的零样本因素残差，不调整特征层、卷积核、训练长度、学习率或损失权重。下一阶段只能在冻结Phase 47源模型基础上使用无标签target-train做一次带源锚和类别占用防塌缩的因素条件适应；其目标不是再做固定置信度硬伪标签，而是检验目标一致性是否能校准三个因素。

<a id="phase-48"></a>

## Phase 48 — 双视图因素条件原型适应（进行中）

### 预注册（结果产生前）

Phase 48冻结Phase 47 iter-1,000模型及全部参数。使用CloudSEN source-train真值在归一化stride-4 mask feature上分别计算阴影/非阴影、有云/clear、厚云/薄云六个源原型；使用全部6,502张无标签target-train图像的原图与水平翻转双视图，只在两个视图二值因素预测一致的像元中，按每图每因素最高10%和最低10%相对分位分别计算目标正/负原型。该相对选择保证每个因素两侧均有候选，不复用Phase 45失败的固定0.90置信度，也不得读取target-train标注。

每个适应原型固定为归一化后的`0.75×源原型+0.25×目标原型`；推理时以温度0.1计算正负原型余弦差，并以固定0.5权重加到Phase 47对应因素logit，再作同一合法四类重构。所有比例、温度与权重在target-val推理前冻结，不做网格搜索。source-val直接沿用不被修改的Phase 47模型73.2179 mIoU，因此遗忘为0。通过线为target-val相对原始V8至少提升1.0 mIoU（≥44.1241）、弱类至少提升2.0（≥17.3059）、宏Boundary F1至少提升1.0（≥14.4763），三个因素的源/目标正负原型计数均非零且target-test、CloudSEN internal test封存。失败即关闭固定混合的离线原型推理，不调整分位、混合比、温度或融合权重，转向可学习但受源锚约束的目标因素adapter。

### 实验结果与止损

完整8,490张source-train与6,502张无标签target-train均处理完成。三个源因素的负/正原型像元计数依次为阴影`126819845/12280315`、有云`76257127/50562718`、厚云条件`13415540/37147178`；目标一致分位计数为`10656779/9871206`、`10356882/10469227`、`9891845/10355064`，六组均非空。适应后1,905张target-val为42.7642 mIoU，clear、cloud shadow、thin cloud、thick cloud IoU分别为75.4462、15.1853、13.2728、67.1526，弱类均值14.2290，宏Boundary F1为13.3652。相对原始V8 source-only分别下降0.3599、1.0769和0.1111点，三项性能门槛全部失败；source模型未改，遗忘为0，target-train标签、target-test和CloudSEN internal test均未读取。

Phase 48失败并关闭固定混合的离线余弦原型推理。原型计数完整排除了空类问题，但源/目标mask feature的传感器偏移不能用固定凸组合和欧氏球面距离消除；按预注册不调整10%分位、0.75/0.25混合、0.1温度或0.5融合权重。下一阶段若继续，只允许训练极小目标因素adapter，并以Phase 47参数为显式L2源锚、以双视图一致性和冻结初始类别占用为防塌缩约束；若该可学习适应仍不能至少超过原始V8，则停止该开发任务的相邻DA变体并回到数据/标签体系审计。

<a id="phase-49"></a>

## Phase 49 — 源锚约束的无标签因素adapter（进行中）

### 预注册（结果产生前）

Phase 49从Phase 47 iter-1,000冻结checkpoint开始，只更新stride-4因素头的387个参数，V8其余参数全部冻结。按seed 49固定顺序遍历6,502张无标签target-train恰好一轮；每张图构造原图与“水平翻转+亮度0.8–1.2+对比度0.8–1.2”的确定性强视图。损失固定为：双视图三个因素概率MSE权重1.0、二元熵权重0.01、相对冻结初始模型的逐图因素占用MSE权重5.0、相对Phase 47因素头参数的L2源锚权重1.0。Adam学习率固定`1e-4`，不使用target标签、不运行target-val选checkpoint，训练结束checkpoint唯一。

固定checkpoint一次性评估source-val与scene-disjoint target-val。通过门槛沿用Phase 48：source-val至少72.98，target-val mIoU至少44.1241、弱类至少17.3059、宏Boundary F1至少14.4763，且目标适应前后三个因素平均占用绝对漂移均不超过0.05、target-test与CloudSEN internal test封存。失败即关闭该可学习因素adapter及Phase 46–49整组相邻DA变体，不调整增强范围、损失权重、学习率或训练轮数，下一步只允许做标签体系/数据协议审计或提出新的非相邻方法假设。

### 实验结果与终止结论

6,502张target-train无标签单轮适应完成，平均总损失0.007304，其中一致性0.004927、熵0.218586、占用MSE `2.16e-05`、参数锚MSE `8.33e-05`。固定最终checkpoint在source-val达到73.2932 mIoU，四类IoU为clear 88.1130、thick cloud 84.6602、thin cloud 59.1216、cloud shadow 61.2780，源域门槛通过且相对Phase 47略升0.0753点。目标三个因素平均占用从`[0.07709, 0.59709, 0.70238]`变为`[0.07407, 0.59964, 0.70073]`，绝对漂移仅`[0.00302, 0.00255, 0.00166]`，远低于0.05防塌缩线。

但1,905张target-val仅为42.8839 mIoU，clear、cloud shadow、thin cloud、thick cloud IoU为75.9967、16.7195、12.0771、66.7422，弱类均值14.3983、宏Boundary F1 13.3133；相对原始V8 source-only分别下降0.2402、0.9076和0.1630点，三项目标门槛全部失败。target-train标签、target-test与CloudSEN internal test均未读取。Phase 49失败并触发终止：Phase 46–49从最终logit因素校准、mask-feature因素头、双视图固定原型到源锚可学习adapter均未产生目标域收益，禁止继续调相邻阈值、权重、增强、学习率或轮数。当前证据支持“Oracle空间充足，但现有V8特征的薄云跨传感器可迁移性不足”；下一研究动作必须回到标签体系/传感器数据审计，或提出能改变共享表征而非只校准输出的全新方法假设。

<a id="phase-50"></a>

## Phase 50 — 1%目标标注主动适应（预注册）

Phase 45的同架构Oracle相对source-only提高13.0963 mIoU，证明任务存在足够监督上限；Phase 46–49的所有无标签因素化DA均未提高目标域，现将失败归类为“Oracle空间大但跨域监督信号失败”。依冻结止损树，不再设计第三套无监督适配模块，只允许一次1%目标标注主动适应可行性实验。

标注预算固定为scene-disjoint target-train的65/6,502张完整512×512图块，即0.9997%，不读取target-val或锁定target-test参与选择。选择前仅使用冻结Phase 22 V8、目标影像及Phase 45清单中的scene/biome元数据：图块效用固定为0.40×弱类（薄云/阴影）响应、0.35×水平翻转及确定性亮度/对比度视图不一致、0.25×预测熵；再以0.50权重的RGB统计与类别响应特征距离做贪心去重。预算按biome样本量最大余数法分配，每个scene最多选择2张，每个biome只在效用前`20×quota`候选中做距离选择。所有权重、候选规模与约束在标签揭示前冻结。

选定65张后才允许读取对应标签；Landsat标签顺序映射回V8的CloudSEN输出顺序。训练从Phase 22 V8 seed-42最佳checkpoint开始，使用全部CloudSEN source-train与重复131次的65张目标图块组成近似1:1源/目标回放，batch 4，AdamW学习率`2e-5`，固定4,000 iter，每1,000 iter仅在开发用target-val选择checkpoint，seed 50。不得依据结果改变选择权重、标注预算、源目标比例、学习率或训练长度。

唯一方向继续线为固定checkpoint在1,905张target-val相对V8 source-only 43.1241至少提高5.0 mIoU，即达到48.1241，同时source-val不低于72.98（遗忘不超过1点）、样本数和封存检查全部通过。弱类mIoU、Boundary F1作为机制诊断报告但不事后增加通过条件。通过后才可转入少样本主动DA，并补做随机等预算、三随机种子和标注成本曲线；未通过立即停止全部适配路线，只允许进行一次可信预测/风险控制可行性审计。target-test与CloudSEN internal test继续封存。

### Phase 50结果与止损

无标签选择完整扫描6,502张target-train后固定选择65张（0.9997%），选择阶段未读取标签；8个biome按预注册比例分配且每scene不超过2张。首次运行在训练前因Phase 49封存字段位置检查失败，第二、三次运行在iter 0前因MMSeg目标到源类别元信息映射检查失败；三次均未产生checkpoint。修复仅校正布尔字段读取以及注册具有CloudSEN类顺序的Landsat映射数据集，未改变冻结选择、标签、预算、训练设置或门槛。修复后固定4,000 iter训练完成，target-val训练期mIoU依次为iter 1,000的48.01、iter 2,000的48.72、iter 3,000的48.13和iter 4,000的48.17，故按预注册选择iter 2,000。

独立逐图汇总中，iter-2,000 checkpoint在1,905张target-val达到48.7111 mIoU，clear、cloud shadow、thin cloud、thick cloud IoU分别为78.8973、16.5136、28.1779、71.2555；弱类均值22.3458、宏Boundary F1 19.7348。相对V8 source-only分别提高5.5870、7.0399和6.2585点，目标侧主门槛及弱类/边界机制指标均明显通过，证明1%有监督目标信号可以纠正纯UDA无法恢复的薄云错误。

但同一checkpoint在535张source-val仅为70.2977 mIoU，四类IoU为87.3995、82.4160、52.3315、59.0438；相对预注册73.98基线遗忘3.6823点，超过最多1点门槛。即使使用近似1:1的完整源域回放，少量目标监督仍通过移动共享表示和分类边界换取目标收益，未形成可接受的双域方法。Phase 50整体失败并执行`stop_adaptation_route_begin_trustworthy_segmentation_audit`：禁止调整源目标比例、学习率、轮数、选择权重或预算，也不进入随机基线、三种子和成本曲线。target-test与CloudSEN internal test均保持封存。下一步只允许一次冻结模型的可信预测/错误定位可行性审计；若现有不确定性不能可靠定位跨传感器错误，则结束TGRS主方法路线。

<a id="phase-51"></a>

## Phase 51 — 跨传感器可信预测可行性审计（预注册）

Phase 50的目标精度、弱类和边界均显著改善但源域遗忘超线，适配路线正式停止。Phase 51不训练第三套模型或uncertainty head，只审计冻结Phase 22 V8 source-only自身的最大归一化Mask2Former语义分数，能否在domain shift下定位错误。仅评估535张source-val与1,905张scene-disjoint target-val；不读取target-train、锁定target-test或CloudSEN internal test。

流式统计全部有效像元的20-bin ECE、Brier score、NLL和risk–coverage/AURC，并以`1−max_probability`作为错误分数报告全体、真值薄云/阴影像元及真实语义边界像元的错误检测AUROC。可靠定位必须同时满足：target全体错误AUROC至少0.70、弱类错误AUROC至少0.65、边界错误AUROC至少0.65、最低置信20%像元召回至少40%的全部错误、保留最高置信80%像元时错误风险相对全覆盖至少下降30%。阈值在推理前冻结，不做温度缩放或target-val后校准。全部通过才允许转入可信分割主线并设计风险控制闭环；任一失败即执行`end_tgrs_main_method_route`，停止当前项目的TGRS主方法扩展，只整理既有工程与负结果论文。

### Phase 51结果与最终退出

首次完整审计在全部535张source-val和1,905张target-val推理结束后，仅因NumPy布尔量不能JSON序列化而在summary落盘前退出；修复仅将既有门槛比较显式转换为Python `bool`，未改变模型、数据、指标、阈值或封存状态，随后原审计完整重跑。冻结V8在source-val的像素accuracy为89.7778%，ECE 0.02035、NLL 0.27660、Brier 0.15037、AURC 0.01745；全体错误AUROC 0.89386、弱类错误AUROC 0.74933，但边界错误AUROC仅0.58071。

跨到target-val后，像素accuracy降至74.2613%，ECE升至0.10369、NLL 0.87852、Brier 0.39390、AURC 0.13347，说明domain shift同时造成明显过置信与风险排序退化。全体错误AUROC为0.73993并通过0.70门槛，最低置信20%像元召回41.8772%的全部错误（lift 2.0939）也通过；但弱类错误AUROC仅0.35064，显著低于0.65且低于随机排序，边界错误AUROC仅0.56353，80% coverage的风险从25.7387%降至18.7001%、相对下降27.3464%，亦低于30%门槛。

Phase 51失败并执行`end_tgrs_main_method_route`。现有置信度能发现一部分一般错误，却对论文核心的薄云/阴影错误呈反向或无效排序，也不能可靠覆盖边界风险；因此禁止继续增加uncertainty head、校准模块、拒识阈值调参或新的适配结构。Phase 45–51的止损树至此闭合：Oracle监督空间真实存在，无监督适配无效，1%主动监督以超过3点源域遗忘换得目标收益，而冻结置信度无法可靠定位弱类/边界错误。target-train未用于Phase 51，target-test与CloudSEN internal test始终封存。当前项目停止TGRS主方法扩展，后续只整理Phase 1–44部署成果、Phase 45无泄漏协议及Phase 46–51系统负结果，转向JSTARS、GRSL、Remote Sensing或工程部署论文。

<a id="phase-52"></a>

## Phase 52 — 稀疏目标传感器 MsRE 可行性（预注册）

Phase 52不是重开Phase 45–51已关闭的输出校准或纯无监督路线，而是检验新的表征级增量假设：Phase 22 DINOv2-S、source Cloud-Adapter和source Mask2Former头全部冻结；目标传感器只启用位于block `[2,5,8,11]`的16-token普通MsRE残差与四尺度rank-8 head-delta。只使用Phase 50在揭示标签前冻结选择的65/6,502张目标图块，训练4,000 iter并每1,000 iter仅以target-val选择checkpoint。target-test和CloudSEN internal test继续封存。

Phase 52A必须同时达到：target-val相对source-only提高至少3.0 mIoU、弱类至少4.0 IoU、宏Boundary F1至少3.0；关闭目标路径是代码级identity bypass，两个独立构建模型的CUDA推理logit最大绝对误差不超过`1e-5`，且完整source-val遗忘不超过0.05 mIoU；每目标新增可训练参数不超过0.50M；同机batch-1、512×512、原生FP16实际延迟增幅不超过20%。任一失败立即执行`stop_phase52_msre_route`，禁止调整token数、注入位置、学习率、轮数或损失，也不得进入元数据门控和云感知token。全部通过才允许Phase 52B比较16/32 tokens与稀疏位置；后续元数据门控还必须在匹配预算下相对普通MsRE提高至少1.0 mIoU或1.5弱类IoU。

### Phase 52A结果与止损

固定4,000 iter训练完成，并按预注册在target-val选择iter-1,000。关闭目标路径后，同一checkpoint在535张source-val为73.9800 mIoU，源域遗忘近似为零；目标新增可训练参数380,577。RTX 4090 D、batch 1、512×512、原生FP16同机测速从冻结基线22.0689 ms增至24.4751 ms，延迟增幅10.90%，参数与20%延迟门均通过。

完整1,905张target-val达到44.8525 mIoU、15.2351弱类mIoU和16.5280宏Boundary F1，相对冻结source-only分别为`+1.7284`、`-0.0708`和`+3.0517`。边界、源域保持、参数、延迟、完整性、有限数值和封存门通过，但目标总体未达到`+3.0`，弱类未达到`+4.0`，其中cloud shadow IoU仅0.8403。Phase 52A失败并执行`stop_phase52_msre_route`：不运行32-token或注入位置消融，不进入传感器元数据门控、云感知token、序数损失或一致性损失。target-test与CloudSEN internal test保持封存。

### Phase 52A事后协议审计：USGS `Shadows?` 字段遗漏

USGS L8 Biome官方逐景目录包含`Shadows? yes/no`字段：96景中仅32景为`yes`。该字段不在下载的MTL文件中，Phase 45 manifest及原Phase 52A均未摄入它。交叉核对官方整景`fixedmask`确认，`no`场景没有shadow truth；因此原Phase 52A把`no`场景class-0像元作为可靠非shadow负样本，并在无shadow真值的场景上计算Shadow IoU，协议不成立。冻结的65张Phase 50选择中只有12张来自`yes`场景、53张来自`no`场景；1,905张target-val中963张/8景为`yes`、942张/8景为`no`。官方网页快照冻结为`research_plans/protocol_data/l8_biome_usgs_shadow_status.csv`，规范LF字节SHA-256为`c735dc14ba5d40ffc0443ba9e5441d2716acd897c80f341a0093319f18862390`，每行保留来源URL。原Phase 52A数值保留用于溯源，但其全场景shadow IoU、弱类均值、Boundary F1及“shadow collapse”结论统一标记为`protocol_contaminated_by_missing_usgs_shadow_status`，不得作为后续机制依据。

<a id="phase-52a-fix"></a>

## Phase 52A-fix — USGS shadow可用性修正版（预注册）

修正版只改变标签有效性：冻结主干/source Cloud-Adapter/source head，保持同一Phase 22初始化、同一65张Phase 50选择、16 tokens、`[2,5,8,11]`注入、rank-8 head-delta、380,577参数、seed 52、batch 4、增强、优化器、学习率、每1,000 iter验证和总4,000 iter完全不变。来自`Shadows?=no`场景的训练图块将L8原始class 0映射为ignore 255，仅保留thin/thick监督；`yes`场景保留完整四类监督。checkpoint选择和目标报告只使用963张、8景的`Shadows?=yes` target-val，并在同一子集重新计算冻结source-only基线；target-test与CloudSEN internal test继续封存。

止损线沿用原相对门槛并增加shadow非回退：相对修正source-only至少`+3.0 mIoU`、`+4.0 thin/shadow mean IoU`、`+3.0 macro Boundary F1`，且shadow IoU增量不得小于0；source-val至少73.93，目标参数必须恰为380,577，同机FP16延迟增幅不超过20%，完整评估535张source-val和963张修正target-val且指标有限。任一失败执行`stop_phase52a_fix_shadow_status_route`，禁止事后调整token、注入位置、损失、采样、学习率或训练长度。

### Phase 52A-fix结果与止损

修正后的4,000 iter训练完成，并在仅含官方`Shadows?=yes`的963张、8景target-val上选择iter-1,000。同一修正子集重新计算的冻结source-only为41.9126 mIoU、19.2148弱类mIoU、14.9421宏Boundary F1、26.7183 shadow IoU和11.7114 thin IoU；候选为43.6728 mIoU、19.2303弱类mIoU和15.7833宏Boundary F1，增量仅`+1.7602/+0.0155/+0.8412`。thin IoU升至37.4599，但shadow IoU降至1.0008，相对修正基线回退25.7175点。

关闭目标路径的source-val为73.9820；目标参数恰为380,577；同机FP16 batch-1延迟从21.4472 ms升至25.0755 ms，增幅16.92%。完整性、有限数值、源域、参数、延迟与封存检查通过，但三个目标增益门和shadow非回退门全部失败。Phase 52A-fix执行`stop_phase52a_fix_shadow_status_route`，不得启动相邻调参或下一阶段实验。该修正结论明确写入官方USGS逐景`Shadows?`字段：排除942张来自`Shadows?=no`、没有shadow真值的target-val图块后，薄云增益与shadow collapse仍然存在，不能再归因于该字段遗漏。

<a id="phase-53"></a>

## Phase 53 — 云影二元辅助监督（预注册）

冻结诊断显示Phase 52A相对source-only的逐类IoU增量为clear `+4.5862`、thick cloud `+2.4689`、thin cloud `+15.4896`、cloud shadow `-15.6312`；thin cloud Boundary F1提高`18.0497`，cloud shadow Boundary F1下降`16.5978`。关闭目标分支的source-val为73.9800 mIoU，确认传感器路径选择实现零遗忘。结论不是“增益全部来自clear/thick”，而是普通MsRE显著改善薄云与薄云边界，却造成云影语义塌缩。

Phase 53只增加一个机制：在最终语义logit上加入正负像元等质量的cloud-shadow二元BCE，固定权重0.5且不增加参数。Phase 52的16个通用token、380,577可训练参数、`[2,5,8,11]`注入位置、rank-8 head-delta、冻结源路径、65张选择、seed 52、优化器、增强、学习率、验证间隔与4,000 iter全部不变；不加入类别token、序数/边界损失、均衡采样、伪标签或元数据门控。

通过线要求target mIoU至少46.1241、弱类mIoU至少19.3059、cloud shadow IoU至少16.4714、cloud shadow Boundary F1至少10.0、thin cloud IoU至少27.6299、关闭目标路径的source-val至少73.93，并保持恰好380,577个目标可训练参数、完整有限评估和两套test封存。任一失败即停止该机制，不调整损失权重、采样、token、位置、训练长度或样本选择。

Phase 53训练期间，用户进一步冻结假设：不得让shadow辅助梯度继续更新已成功的Phase 52A MsRE，而应冻结Phase 52A全部参数，只训练独立shadow/non-shadow residual。由于Phase 53不再隔离所需机制，任务在完成和checkpoint评估前终止；中途训练值不作为实验结果。后续先完成Phase 52A云影去向、预测占用、1%标注覆盖和特征原型距离诊断，再预注册冻结residual实验。

<a id="phase-52b"></a>

## Phase 52B — 冻结薄云路径的云影关系残差（预注册）

四项冻结诊断确认类别塌缩：Phase 52A把80.4974%的真值shadow判为clear、14.3310%判为thin、4.2550%判为thick，仅0.9166%仍为shadow；预测shadow占比从source-only的6.3531%降至0.2820%。65张标注仅含50,572个shadow像元（0.2968%），只有8张含shadow，覆盖6/8个biome。Phase 52中后层shadow原型到thin、非零辐射暗clear代理、thick和全部clear的余弦距离依次为0.0769、0.0866、0.1603和0.1663；数据没有水体或山体阴影独立标签，因此不作对应因果声称。

Phase 52B冻结iter-1,000 Phase 52A的主干、source Cloud-Adapter、MsRE、target head-delta和Mask2Former，只训练零初始化shadow residual。分支读取output-index-2特征、冻结四类概率与RGB亮度，使用389→32点卷积、dilation-3的32通道depthwise 3×3和32→1点卷积；只修改shadow log-odds，严格保持clear/thin/thick内部比例。训练仍为同一65张、seed 52和4,000 iter，每batch 4保证一张shadow-positive图；固定loss为balanced BCE 1.0、Tversky 1.0和一像素shadow boundary BCE 0.5。

门槛为零初始化概率误差不超过`1e-5`且预测完全一致、新增参数不超过0.1M、target mIoU至少48.0、shadow IoU至少16.471、thin IoU至少28.630、thin Boundary F1至少19.934、关闭全部目标路径后source-val至少73.93，以及完整有限评估与两套test封存。任一失败执行`stop_shadow_residual_route`，不得事后调整loss、batch组成、宽度、dilation、训练长度或门槛。

### Phase 52B结果与止损

固定4,000 iter完成并选择iter-3,000。首次评估在第一张source图像前因自定义包装器遗漏原FP16 autocast而停止，修复仅恢复评估精度上下文，未改变checkpoint、模型、数据或门槛，随后完整重跑。目标域达到43.9302 mIoU；shadow IoU从Phase 52A的0.8403恢复到8.3945，但仍低于16.471，thin IoU从29.6299降至23.5214，thin Boundary F1从20.9335降至15.8784，shadow Boundary F1为9.3660。源域、参数、完整性、有限数值及封存门通过，但全部目标性能门失败。

Phase 52B执行`stop_shadow_residual_route`。结果表明分支能部分恢复shadow，但即使数学上固定non-shadow类内部比例，提高shadow概率仍会从原本正确的thin像元中夺取预测；禁止继续调整loss、阈值、宽度、dilation、采样或训练长度。

<a id="phase-55"></a>

## Phase 55 — thin/shadow非对称负迁移机制审计（预注册）

Phase 52A-fix在修正的`Shadows?=yes`子集上使thin IoU从11.7114升至37.4599（`+25.7485`），同时使shadow IoU从26.7183降至1.0008（`-25.7175`），弱类均值仅变化`+0.0155`。Phase 55不训练新分割模块，只验证这种近似等量交换是否确由表征级梯度冲突造成。source-only固定为Phase 22 iter-40,000，候选固定为Phase 52A-fix iter-1,000；诊断拟合只使用冻结65张选择中官方`Shadows?=yes`的12张，线性探针只在963张、8景`Shadows?=yes` target-val上作scene-disjoint评估，两套test继续封存。

审计A在候选最终checkpoint分别构造thin与shadow的像素平均负对数语义概率损失，只使用得分分辨率下同时含两类的训练图块；在`[2,5,8,11]`四个注入层测量post-MsRE token梯度和逐层learnable-token梯度的余弦及seed-55 bootstrap 95%区间。冲突门为至少2/4层的pooled post-MsRE余弦小于`-0.10`，且四层均值小于0。审计B对source-only和候选的全部12层、source Cloud-Adapter之后及候选MsRE之后的384维patch feature，使用固定`lambda=0.01`类别均衡ridge probe；候选必须在至少一个注入层shadow AUROC低于source至少0.05，且第11层仍未恢复到距source 0.02以内。审计C将相同类别损失对全部380,577目标参数的逐图归一化梯度定义为thin更新与shadow经验Fisher/Jacobian子空间，固定保留90%奇异值能量、rank最多8；最小主夹角必须不大于75度、shadow梯度投影能量至少10%，且聚合thin/shadow梯度余弦为负。

只有三门全部通过，才允许把机制结论写为“thin/shadow表征级梯度冲突”并预注册解耦模块；任一失败执行`stop_thin_shadow_conflict_hypothesis`，只保留非对称负迁移现象，不得事后扫描阈值、rank、probe、正则项或由此设计冲突专用结构。

### Phase 55结果与止损

完整审计在RTX 4090 D上使用实际存在且与Phase 52A-fix相同的`cloud-lite-pt210`环境完成；训练机不存在请求名称`cloud-lite-210`。12张官方`Shadows?=yes`训练图块中，得分分辨率下同时含thin和shadow的5张进入梯度审计；线性探针完整评估963张、8景修正target-val。结果文件SHA-256为`ea3c3566487472d266d0f060eed1b1b0e53d3321af0551e84a314bdde7f71c83`。

四个注入层`[2,5,8,11]`的pooled post-MsRE thin/shadow梯度余弦依次为`+0.1628/+0.3860/+0.2518/-0.0018`，均值`+0.1997`，没有至少两层低于`-0.10`。虽然第11层逐图余弦均值为`-0.1233`、bootstrap 95%区间`[-0.1659,-0.0711]`，但等权聚合梯度后几乎为0，冲突门失败。四个注入层的shadow probe AUROC候选相对source差值为`-0.0006/-0.0295/+0.0254/+0.0425`；最终第11层候选为0.6383，高于source-only的0.5958，表明shadow线性信息没有在MsRE逐层消失，表征损失门失败。thin与shadow参数梯度子空间均为rank 3，主夹角`20.82/64.31/86.62`度，shadow梯度能量有43.91%落入thin子空间，但聚合有符号余弦为`+0.4337`而非负值，子空间冲突门也失败。

Phase 55三门全部失败并执行`stop_thin_shadow_conflict_hypothesis`。Phase 52A-fix的近似等量IoU交换仍是成立的非对称负迁移现象，但证据否定“MsRE抹除shadow信息”及“thin/shadow更新全局对抗”这两个机制解释；现象更可能发生在仍含shadow信息的表征之后、共享head/readout的类别竞争中，但该表述仅为定位性推断，不授权继续设计冲突解耦模块。target-test与CloudSEN internal test保持封存。



---

<a id="phase-56-59"></a>

## Phase 56–59 实验记录

### Phase 56（已登记，待运行）

- 目标：在严格分离的 target-train/target-val 缓存上，验证冻结 Phase 52A 特征能否由参数受限 readout 同时恢复 shadow 且保留 thin。
- 改动：新增确定性分层特征缓存、readout/共现审计、SHA256 manifest、单 GPU 注册表与锁；L8 Biome 只使用 USGS `Shadows?=yes`，65 张冻结选择中实际可拟合 12 张，验证集 963 张。
- 网络：数据与 checkpoint 均使用远端已有文件；训练过程不需要外网，代码仅通过 Git SSH 同步。
- 止损线：Shadow IoU `>=13.86`、Thin IoU `>=32.31`、相对 Phase 52A-fix target mIoU `>=+2.0`，三项必须同时满足；失败则不运行 Phase 57。
- 结果：尚未运行，不报告数值。

#### 启动事件 1

- 目标：在 `gzs` 后台启动 Phase 56 全流程。
- 改动：无实验参数改动；启动器提交为 `0f1c2c3` 后已按 Git push/pull 同步。
- 网络：Git SSH 正常；数据未联网下载。
- 止损线：尚未进入科学门槛判断。
- 结果：失败于实验启动前；`nohup` 非交互 shell 报 `conda: command not found`，未创建 registry、未提取缓存、无指标。后续仅修复 conda 初始化路径后重启。

#### Phase 56 完成记录

- 目标：判断冻结 Phase 52A 特征上的参数受限 readout，能否同时恢复至少一半 shadow 损失、保留 80% thin 增益并带来 `>=2.0` target mIoU 提升。
- 改动：提交 `70e1088` 仅修复后台 conda 初始化；完整 run 使用提交 `70e1088f1fa42064d16209a2878549132590c2a6`。模型权重未训练；使用 target-train scene 级三折 CV 选择 readout，再在完整 963 张 target-val 上密集评估。65 张冻结选择中只使用 `Shadows?=yes` 的 12 张，排除 53 张 `Shadows?=no`；其中 thin+shadow 共现 7 张。
- 网络情况：冻结 Phase 52A-fix 网络与 source-only 网络仅作特征提取；数据/checkpoint 全部来自远端已有文件，外网只用于 Git SSH push/pull，无数据下载。target-test 与 CloudSEN internal test 均未读取。
- 止损线：Shadow IoU `>=13.86`、Thin IoU `>=32.31`、target mIoU 相对 Phase 52A-fix `>=+2.0` 必须同时满足；任一失败即停止本体树主线。
- 结果：`P56_readout_MSRE_CS2L8_cache_s56` 完成。target-train CV 选择 `pixel_decoder_mask + cosine`（CV mIoU `38.8749`）。全像素 target-val：mIoU `43.2785`、clear/thick/thin/shadow IoU=`76.4923/59.9695/17.3436/19.3087`，shadow recall `32.1866`，ECE `0.31244`，NLL `0.97025`。Shadow 门通过；Thin 门失败；相对 Phase 52A-fix mIoU `43.6728` 为 `-0.3942`，增益门失败。总门失败，决策为 `stop_readout_and_ontology_mainline`，不启动 Phase 57–59。汇总 SHA256：`c2e3cfc19596843dae921188dd47b6e5ebc7ae84b66b23e8a04a5d193deda4dd`。

### Phase 57（止损封存，未启动）

- 目标/改动/网络情况/结果：未开展；Phase 56 前置门失败。`src/tools` 仅保留硬门禁脚本，不实现或运行层次 head。
- 止损线：Phase 56 `passed=true`；当前为 false。

### Phase 58（止损封存，未启动）

- 目标/改动/网络情况/结果：未开展；Phase 57 不具备启动资格。`src/tools` 仅保留硬门禁脚本。
- 止损线：Phase 57A/57B 均通过；当前无合格结果。

### Phase 59（止损封存，未启动）

- 目标/改动/网络情况/结果：未开展；Phase 58 不具备启动资格。`src/tools` 仅保留硬门禁脚本。
- 止损线：Phase 56、57A、57B、58 连续通过并冻结最终方法；当前不满足。


---

<a id="phase-60"></a>

## Phase 60 实验记录

### 2026-09-24：协议冻结与启动

- 目标：停止新增分割模型，独立审计多光谱信息充分性（60A）与目标标注支持上界（60B）。
- 改动：新增 `tools/run_phase60a_multispectral_audit.py`、`tools/prepare_phase60b_support_audit.py`、`tools/evaluate_phase60b_support_audit.py` 和远程串行入口 `tools/run_phase60_4090d.sh`。
- 网络/读出：60A 不运行分割网络，只使用 Phase 54 已对齐缓存和固定 class-balanced ridge；60B 冻结 Phase 22 source 网络做非 oracle acquisition，冻结 Phase 52 adapted 网络提取 `pixel_decoder_mask`，仅拟合同一个 class-balanced ridge。Oracle 只作上界，禁止进入最终方法。
- 数据口径：只使用 USGS `Shadows?=yes`。合法 target-train 池为 1560 张/18 scenes，target-val 为 963 张/8 scenes；Phase 50 原 65 张中仅 12 张合法。指定预算 16/32/65/130/325 不变，但占合法池比例为约 1.03%/2.05%/4.17%/8.33%/20.83%，不伪称 0.25%–5%。
- 60A 输入：S0 RGB；S1 +NIR；S2 +SWIR1；S3 +SWIR2；S4 六公共通道加预注册 NDVI/NDMI/NDSI/NBR。S0–S4 使用同一批像素（每图每类最多 32）。
- 范围限制：授权目录内只有 RGB/NIR/SWIR1/SWIR2 对齐缓存；Landsat 全 11 波段原始档案位于授权目录外，因此 S5 明确记为 blocked，不生成替代数据或结果。
- 止损线：60A 同一 readout 下 Thin IoU ≥32.31、Shadow IoU ≥20、mIoU 相对 S0 ≥2，且必须由公共波段条件达到；60B 按 65/130/325 的 oracle 阈值判读采样、标注量或观测/标签瓶颈。
- 结果：尚未产生；不得提前填写。

### 2026-09-24：60A 完成

- 目标：判断固定浅层读出下，公共多光谱通道能否同时补足 thin/shadow 并相对 RGB 提升至少 2 mIoU。
- 改动：无训练中改码；按冻结协议完成 S0–S4，S5 因全 11 波段原始档案不在授权目录内保持 blocked。
- 网络/读出：未训练或微调分割网络；12 张合法 Phase 50 target-train 图块、每图每类最多 32 像素、同一个 class-balanced ridge；963 张/8 scenes 合法 target-val。
- 止损线：Thin ≥32.31、Shadow ≥20、mIoU 相对 S0 ≥2，且公共波段条件达到全部三项。
- 结果（dense 32×32 readout 后最近邻回投）：S0 mIoU/Thin/Shadow/H/AUROC=18.853/3.893/0.386/0.703/0.4172；S1=21.957/9.430/0.478/0.910/0.2887；S2=20.932/1.033/2.333/1.432/0.5172；S3=21.527/5.047/3.627/4.221/0.5626；S4=23.783/12.095/7.602/9.336/0.6413。AUROC 为同像素 shadow-vs-thin sampled probe。
- 稳定性：S4 scene-bootstrap 95% CI：mIoU [14.665,29.779]、Thin [1.186,21.200]、Shadow [2.454,11.457]、弱类调和均值 [2.046,11.921]；各 biome 的 Thin/Shadow 波动明显，公共波段提升不稳定。
- 判定：S1–S4 虽均达到相对 S0 的 +2 mIoU，但均未达到 Thin/Shadow 双阈值，60A `stop_multispectral_line`。结果 SHA256：`64987af34d92c3699b2279efa88c51885f91d1ea72fe7f9c3647140c9e5d8e2a`。

### 2026-09-24：60B 完成

- 目标：用完整 target-train 标签模拟主动学习，判断 65 张失败究竟来自采样还是标注量/观测与标签体系。
- 改动：在 1560 张合法池上完成 random、entropy、core-set、uncertainty+diversity、Phase50-policy-rebased、oracle class-balanced、oracle thin-shadow relation；原 Phase 50 单列为 nominal-65/实际合法 12。Oracle 只作上界。
- 网络/读出：冻结 Phase 22 source 网络计算非 oracle acquisition；冻结 Phase 52 adapted `pixel_decoder_mask`；所有预算使用 seed=56、每图每类最多 32 像素和同一个 class-balanced ridge；dense 验证固定 963 张/8 scenes。
- 止损线：65 张 oracle 同时达到 Thin ≥32、Shadow ≥20 才可判采样瓶颈；若仅 130/325 达到则否定“1%足够”；325 仍失败则归因更根本的数据/标签/传感器可观测性问题。
- 结果（mIoU/Thin/Shadow/H）：原 Phase50 nominal-65（合法 12）=43.245/16.406/18.603/17.435；65 张 random=47.081/26.852/27.749/27.293，Phase50-rebased=44.722/23.608/23.576/23.592，entropy=42.275/15.713/17.114/16.384，core-set=45.076/25.734/20.335/22.718，uncertainty+diversity=41.918/17.036/17.246/17.141，oracle class-balanced=43.324/21.101/25.065/22.913，oracle relation=43.042/25.577/23.514/24.502。
- 上界：325 张 oracle class-balanced=46.490/27.021/27.486/27.252；oracle relation=47.705/28.355/27.517/27.930。两者 Shadow 过线但 Thin 仍低于 32；不存在首次达标预算。
- 判定：`no_joint_solution_under_current_protocol`。不能宣称 1% 标注足够，也不能把 oracle 纳入最终方法；仅靠更换当前采样策略不足以跨过双弱类门槛。
- 产物：selection SHA256 `674b644c7d4fbe46763ba924ff09f36a54f4544337714bab85e298756524ad3d`；60B summary SHA256 `623edacc8808e2f1e5ec2818bb02e0ae34a975d5b1ab65d34bcb65a1517da89b`；总摘要 SHA256 `706e2d6adf6cd9a2ffe33b33e33f949c6f93f5d2e375b644912552a07d64049c`。

### 2026-09-26：Phase 61–62后的解释边界修正

- 60B中的`oracle`仅指使用已知目标标签构造的label-aware选择规则，不是目标任务的模型性能上界；它不能进入最终方法，也不能据此证明增加标注必然无效。
- 所有预算使用冻结Phase 52特征与同一浅层ridge readout，因此325张仍未越过Thin/Shadow联合门，只能否定“在当前特征、读出、标签口径和训练协议下，仅增加当前形式的支持样本即可解决问题”。
- S5全11波段被阻塞，target-only完整训练在历史Phase 45又曾达到Thin 39.4668、Shadow 24.4730；因此Phase 60不得表述为“Landsat传感器根本不可观测thin/shadow”。
- Phase 61进一步确认原nominal-65只有12张shadow-valid完整监督；Phase 62确认当前`{clear,shadow}`集合式部分标签也不能补足这一信息缺口。最终冻结判定保持`no_joint_solution_under_current_protocol`，不作更强因果外推。


---

<a id="phase-61"></a>

## Phase 61 实验记录

### 2026-09-24：协议冻结与启动

- 目标：审计 CloudSEN12/L8 Biome 的标签血缘、优化充分性和标签粒度；准备独立双盲人工复核。
- 改动：新增 61A 血缘表、61B 固定 4000-step 浅层优化、61C 同预测标签折叠、61D 盲化复核包及评分脚本。
- 网络/读出：不训练分割网络。61B 只在冻结 Phase52 `pixel_decoder_mask` 上训练线性 softmax readout；61C 只重算既有 confusion；61D 只做冻结推理和 RGB crop。
- 数据口径：L8 Biome 仅 `Shadows?=yes` 可作为完整四类标签；Phase52A-fix 的 53 张 `no` 图块仅保留 thin/thick 有效像元，raw target ID 0 为 ignore。
- 止损线：61A 任一名称集合/hash/路径/mask/mapping 血缘不清即暂停 61B–D；61D 在两位独立人员完成前不得报告 κ，也不得发表“标签体系不一致”的核心结论。
- 结果：尚未产生；不得提前填写。

### 2026-09-25：61A 协议血缘审计完成

- 目标：核实 Phase50/52A-fix/55/56/60 是否共享 manifest、选择集、类别映射与 checkpoint，并定位 53 张样本的排除原因。
- 改动：无训练；逐项读取 manifest、selection、USGS metadata、mask 和既有 cache metadata，生成血缘 JSON/Markdown 表。
- 网络/读出：Phase50 使用 source checkpoint `d33a81337544…`；Phase52A-fix 及后续 adapted 分支使用 `9655ee83c638…`；manifest hash 均为 `885c2d7f61ca…`。
- 止损线：任一名称集合/hash/路径/mask合法值/映射不一致即暂停后续。
- 结果：全部硬门通过。Phase50 对 6502 张 RGB 打分并选同一组 65 张，不读取标签；Phase52A-fix 实际读取这 65 张，其中 53 张 `Shadows?=no` 的 raw target ID 0 被 ignore，仍有 9,987,543 个有效监督像元；Phase55/56/60A 则整图过滤这 53 张，读取同一组合法 12 张（3,145,728 像元）。
- 排除原因：53/53 均仅因 scene 级 `Shadows?=no`；不存在路径缺失、mask 非法值、类别映射或 split 错误，也没有阶段暗换另一组 65 张。
- 产物：lineage JSON SHA256 `d7a5a6317a173977085c4f4fc719a091649886f4755d26c693a582c57547016b`；表 SHA256 `0d52b386643a417ef2cf99da0811ebcbfc946604b0df926dbfe03fb113c763bf`。判定 `lineage_clear_continue_61b_61c`。

### 2026-09-25：61B 优化充分性审计完成

- 目标：区分优化不足、训练支持不足和验证泛化/标签异质性。
- 改动：冻结 Phase52 `pixel_decoder_mask`，对 36 个方法×预算组合使用同一线性 softmax、class-balanced CE、AdamW、4000 steps、每 step 4 张图；不更新分割网络。
- 网络/采样：有效像素比例 source/target=0/100%；16/32/65/130/325 的平均每图曝光为 1000/500/246.154/123.077/49.231 次。
- 止损线：train Thin 低指向优化/表示能力；train 高而 val 低指向覆盖/异质性；325 未收敛才可归因曝光不公平。
- 结果：所有 325 组均满足 loss 收敛判据，故 325 失败不能归因于曝光不足。Random 的 train→val Thin/Shadow：16 为 50.67/70.98→28.80/51.05，32 为 43.40/63.02→30.68/51.86，65 为 44.04/58.44→30.56/55.28，130 为 42.44/57.68→33.09/54.73，325 为 40.37/57.32→33.33/56.00。
- 关键诊断：原 Phase50 nominal-65/合法12 的 train Thin/Shadow=65.70/67.54，val=21.97/29.97，存在显著 train–val 断层。Random65 并未稳定优于 label-aware；Phase50-rebased@65 的 val Thin/Shadow=32.91/52.40。单 seed 不允许使用“稳定优于”措辞。
- 产物：SHA256 `e4b2da63bd9decc265f89a6d1454c87980296f582e9168985a795af55baff176`。

### 2026-09-25：61C 标签粒度坍缩完成

- 目标：对完全相同的四类 confusion 仅折叠标签 ID，测量四类→三类→二类的性能恢复。
- 改动：无推理重跑、无训练；三类为 clear/cloud(thin+thick)/shadow，二类为 cloud/non-cloud，另报 clear/contaminated。
- 结果（四类/三类/cloud-noncloud/clear-contaminated mIoU）：Phase52 source-only=41.91/58.36/76.39/74.38；Phase52 adapted=43.67/48.92/75.13/71.51；Phase56 readout=43.28/55.24/76.42/71.64；Phase60 random65=47.08/57.33/76.89/70.42；oracle325=47.70/56.91/76.22/70.43。
- 判定：粗粒度任务大幅恢复，支持主要域差异包含 thin/thick/shadow 边界语义；但这只是计算证据，61D 独立复核完成前不得发表“标签体系不一致”的核心结论。
- 产物：SHA256 `9d1c09593d3ed009a30d2fab82dbc0076660eb4d88780f29afd58e0eabf64ee3`。

### 2026-09-25：61D 双盲复核包准备完成

- 目标：为至少两位独立人员提供不知道原始标签、模型结果和抽样来源的复核材料。
- 改动：冻结 Phase52/Phase56 预测，从 `Phase52 pred thin/GT shadow` 与 `Phase56 pred shadow/GT thin` 两类错配中，按 biome、S4 高/低置信度、boundary/interior 分层生成 128×128 RGB crop；生成封存映射、说明、两份独立空白 CSV 和评分脚本。
- 网络/读出：仅冻结推理；不训练或修改任何网络。候选 7,884 个，最终盲化单元 200 个，图像文件 200 个；reviewer_A/B 均为表头加 200 条空白记录。
- 止损线：两份独立 CSV 完整锁定前，Cohen/Fleiss κ、共识 IoU、thin/shadow 一致率、原标签一致率和 biome 一致性均不得报告。
- 结果：状态 `awaiting_two_independent_human_reviews`；`human_agreement_metrics_available=false`。sealed manifest SHA256 `8c774bc2f121459fe780654fbae7178b78e1113446d51b2a7ca6a4ebf74a66fd`；packet summary SHA256 `0d5b6aa87ae5ebc52ac230d662e04ceb567877d56c7f36dfc8d6bfd6d6118f49`；Phase61 总摘要 SHA256 `3018bb09e3c266a464fa696ab4495070088d9fcc1144d31df18a18ad4ed59c05`。
- 判定：计算审计完成，但 Phase61 整体尚未完成；在两位人员复核前，只能说“标签粒度不匹配假说获得计算支持”，不能发表“两个数据集标签体系不一致”的核心结论。

### 2026-09-25：61D 双盲复核首次评分（缺失值临时审计）

- 目标：导入 reviewer A/B 的独立复核结果，计算一致性、类别交并比、thin/shadow 一致率、原标签一致率及 biome 分层稳定性。
- 改动：将 `/mnt/i/zhc/reviewer_A.csv` 与 `reviewer_B.csv` 原样复制到远端 `src/work_dirs/phase61/blind_review/completed_reviews/`，不删除或改写源文件；评分器新增严格缺失校验、complete-case 临时模式、输入 SHA/缺失 tile 留痕，以及唯一无歧义的拼写规范化 `ambiguous haze/cirru`→`ambiguous haze/cirrus`（1 条，不作语义代填）。
- 网络/读出：不训练、不推理、不修改网络；在 `cloud-lite-pt210` 中只运行统计评分。本地提交 `7ef9da8` 已 push；gzs 首次 pull 遇到 GitHub GnuTLS 握手失败，第二次重试成功并 fast-forward 到同一提交。
- 数据完整性：sealed 200 条；A 完成 193、缺 7，B 完成 198、缺 2；双方缺失重叠 1 条，因此双人 complete-case 为 192，排除 8 条。A/B 原始 SHA256 分别为 `d5890697dafa64fe8b4fc77fa244b6ce244ac8be1d45d8be9b2c30b4802d3477`、`931b28ac4c82034b29adb85ad106cb3e357815e8335094448e9f280fd68ae440`。
- 止损线：状态固定为 `incomplete_preliminary`；不得把 192 条临时结果冒充完整 200 条正式结论，不得代填 9 个 reviewer 空单元，也不得据此发表“标签体系不一致”的核心结论。
- 结果（192 个双人有效单元）：Cohen κ=0.1363，Fleiss κ=0.1286，精确一致率=28.65%。reviewer 间 IoU：clear 41.56、thin 8.06、thick 5.88、cloud-shadow 19.35、terrain/water-shadow 0.00、ambiguous-haze/cirrus 7.84、uncertain-boundary 9.84（%）。原标签均为 thin/shadow 的抽样单元上精确一致率同为 28.65%；仅在双方都选 definite thin/shadow 的 17 条中，thin-vs-shadow 一致率为 64.71%。
- 共识/原标签：两人完全一致才形成共识，共 55/192（28.65%）；原标签与共识一致率 12.73%，Thin/Shadow IoU 分别为 14.29/10.53。该错配抽样设计没有 original clear/thick 支持，因此 clear/thick 的原标签 IoU 不作解释。
- biome 一致性（双人精确一致率）：barren 15.00%、forest 33.33%、grass_crops 36.11%、shrubland 27.78%、snow_ice 31.03%、wetlands 30.56%；各 biome 均未显示高一致性，但缺失主要集中在 snow_ice（29/36 complete-case），只能作为临时诊断。
- 产物：`review_metrics_preliminary.json` SHA256 `d24112930f1fa39c8c1b80d9ef5e32a2b3cafee55c08d38ad765c8ab861920db`；结构与状态断言复核通过。

### 2026-09-25：61D 空值语义确认与正式封账

- 目标：依据复核方说明，将 CSV 空值从“漏标”更正为“图像纯色、无有效信息，无法评估”的 reviewer abstention，并完成正式 61D 统计。
- 改动：评分器新增显式 `unassessable_solid_image` 空值策略；非法标签仍失败，普通漏标仍只能生成临时结果。正式运行同时更新 blind-review packet 与 Phase61 总摘要，保留每位 reviewer 的空值 tile、原始 SHA 和 1 条无歧义拼写规范化记录。
- 网络/读出：不训练、不推理、不修改网络；本地提交 `cd6061e`、`03fe36d` 均已 push，gzs 正常 pull 后在 `cloud-lite-pt210` 环境运行统计与摘要封账。
- 数据口径：sealed 200 个单元；8 个单元因至少一位 reviewer 判定纯色无信息而排除成对统计，剩余 192 个共同可评单元。9 个 reviewer 空单元中有 1 个双方重叠；空值不是类别，也不参与 κ、IoU 或一致率分母。
- 止损线：不得把不可评空值映射为 clear/uncertain 等类别；结论仅适用于从两类模型错配中分层抽取的 192 个可评位置，不能外推为全数据集随机样本的总体发生率。
- 结果：正式 Cohen κ=0.1363，Fleiss κ=0.1286，双人精确一致率=28.65%；reviewer 间 clear/thin/thick/cloud-shadow/terrain-water-shadow/haze-cirrus/boundary IoU 分别为 41.56/8.06/5.88/19.35/0.00/7.84/9.84（%）。双方均使用 definite thin/shadow 的 17 条中二分类一致率为 64.71%。
- 原标签/共识：55/192 形成双人共识，共识与原标签一致率 12.73%，Thin/Shadow IoU=14.29/10.53；A/B 各自与原标签精确一致率为 24.87%/11.11%。biome 双人一致率为 barren 15.00%、forest 33.33%、grass_crops 36.11%、shrubland 27.78%、snow_ice 31.03%、wetlands 30.56%。
- 判定：Phase61 完成。独立复核为“选定错配位置存在显著 thin/shadow/边界语义歧义及协议不兼容”提供证据；但由于样本按模型错配条件抽取，不能单独证明两个数据集全局标签体系均不一致。
- 正式产物：`review_metrics.json` SHA256 `19002ad19c80a899e094b0b994fd9c87ab8ed5efd1819b9798d5e27fc70d20d6`；`packet_summary.json` SHA256 `a9f3b70744e3e8304929a59923bf2411a4349f9c23eca3651063145d11c7105a`；`phase61_summary.json` SHA256 `2cc5f993cad471d4d25648d8e10c8f71bbcac58007aca417f9f9e65315a824d7`，状态 `phase61_complete`。

### 2026-09-26：61D科学解释边界

- 本次200个盲化单元不是从L8 Biome总体随机抽样，而是从`Phase52 pred thin/GT shadow`与`Phase56 pred shadow/GT thin`两类模型错配中分层抽取；因此所有比例只描述高风险错配位置，不是全数据集错误率或歧义率估计。
- Cohen/Fleiss κ约0.13同时伴随28.65%的精确一致率和较低逐类IoU，说明分歧不只是κ的类别不均衡效应；但reviewer自身一致性不足，也意味着55条双人共识不能直接充当完整四类金标准。
- 严禁将“共识与原标签一致率12.73%”改写为“87.27%原标签错误”。前者只在55/192条形成双人共识的条件子集上定义，且抽样本身富集了模型错配。
- 当前可发表级结论限于：被模型错配条件选中的位置存在显著thin/shadow/边界判读歧义，并提示现有细粒度标签协议与可用影像证据之间可能不兼容。它不能单独证明CloudSEN12与L8 Biome在全局上具有不同标签本体。
- 若后续把人工复核用于主要论文证据，应另设校准集统一操作定义，在保持对原标签和模型预测盲态的同时提供任务所需的全部波段、推荐合成和必要太阳几何信息；允许`uncertain`/集合标签，并用新的代表性或确认性样本报告复核后的一致性。


---

<a id="phase-61d2"></a>

## Phase 61D2 实验记录

### 2026-09-26：校准协议与执行前审计

- 目标：重做校准后的盲法复核；reviewer 对原标签、模型和方法盲态，但获得 B1–B11、推荐合成、太阳几何和必要元数据；40 个校准单元不进入统计，160 个独立单元与 50 个未见确认单元进入主复核。
- 改动：新增 61D2 复核包生成、校准锁、分歧专用第三方裁决和评分脚本；支持 `uncertain`、`unobservable_nodata` 与最多两个语义类的集合标签；硬门按 κ/AC1、Thin/Shadow reviewer IoU 和原标签—裁决标签 IoU 自动执行。
- 网络/读出：不训练、不修改任何模型；只做确定性抽样、11 波段证据页渲染和人工复核统计。确认集不按模型错误或置信度筛选。
- 数据口径：旧 200 个错配分层单元中确定性抽取 40 个校准单元，剩余 160 个作为独立复核；另从旧包未出现的 `Shadows?=yes` target-val 图块按 Thin/Shadow、scene/biome、boundary/interior 平衡抽取 50 个确认单元。校准集被评分器结构性排除。
- 止损线：校准锁定前不得开始主复核；A/B 原始判断必须保留，第三位专家只裁决真实分歧；κ 或 AC1 任一低于 0.40 时四类 mIoU 不得作为主指标；Thin 或 Shadow reviewer IoU 任一低于 0.30 时终止四类硬标签主线；不得以旧 RGB 图或 Phase54 三个辅助光谱通道冒充 B1–B11。
- 真实状态：本地 4 项协议单元测试、语法检查和 11 波段证据页合成渲染 smoke test 通过。远端仓库内 `data` 为指向 `/home/scv/shared/data` 的符号链接；完整 B1–B11/MTL 不在已授权的 `/home/scv/Cloud-Adapter-light` 实体目录内，现有 Phase54 cache 也不含全部波段。因此尚未读取外部 raw-root、尚未生成正式复核包、尚无人工结果或 κ/AC1/IoU；等待明确只读授权或将原始数据放入授权目录。

### 2026-09-26：只读授权后生成正式复核包

- 目标：在不泄露原标签、模型输出、方法或抽样来源的前提下，为两位 reviewer 提供完成校准和独立复核所需的完整多光谱证据。
- 改动：用户明确授权只读访问 `/home/scv/shared/data/l8_biome_raw`；生成过程只写 `/home/scv/Cloud-Adapter-light/src/work_dirs/phase61d2_calibrated_review`。原始目录含 96 个 scene 主栈，每个主栈 11 波段，未尝试写入。
- 网络/同步：本地提交 `96759c5` 已 push；gzs 最初连续出现 GitHub GnuTLS/SSH 连接失败，本轮重试后成功 fast-forward 至同一提交。远端使用 `cloud-lite-pt210`，4 项协议测试通过；不训练或修改模型。
- 数据与产物：生成 40 个校准单元、160 个独立单元和 50 个确认单元，共 250 张证据页；确认集使用 50 张互不重复且未出现在旧 61D 包中的图块，原始 Thin/Shadow 各 25。主复核 210 个单元的 biome 数为 snow_ice 44、shrubland 22、grass_crops 40、barren 36、wetlands 38、forest 30。A/B 的校准和主复核 CSV 均保持全空白。
- 质量核验：250/250 PNG 均为 1040×1210 RGB，ID 与 sealed manifest 一一对应；三张跨校准/主复核抽样证据页目视通过；ZIP 有 257 个 reviewer 可见成员和 250 张 PNG，不含 sealed manifest。ZIP SHA256 `df772757afedebab0a82599d59a10ad51bbd2e7b3858721f4862cf0fcb04af49`；sealed manifest SHA256 `7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77`；packet summary SHA256 `0c2f36c49ceeea87b43549d83c0677138adfdd6714b3edc37fe965a4873ea402`；manual SHA256 `cfa8ec5b65474dc4e6d4402cc9db0838db237afd715f081ff88f41cdb068d172`。
- 止损线与结果：状态为 `awaiting_calibration_lock`，`human_results_available=false`。40 个校准单元完成共同讨论并锁定前不得开始主复核；当前没有人工标签、κ、AC1、reviewer IoU、原标签—裁决标签 IoU或四类主线结论，严禁提前填报。

### 2026-09-28：校准文件封存与锁定

- 目标：核验两位 reviewer 的校准提交并锁定统一协议，校准单元不得进入主统计。
- 改动：将本地提交的 `reviewer_A_calibration.csv`、`reviewer_B_calibration.csv`、`calibration_consensus.csv` 原样复制到远端仓库 `completed_reviews/`，未覆盖 reviewer 模板；校准锁脚本同步更新 packet 状态。因共识文件未提供文字版 `rule_or_counterexample`，锁中如实记录“40 个共识标签已提交、无文字规则修订，手册 v1.0 保持不变”，不补写讨论内容。
- 网络/读出：不训练、不推理、不读取 sealed 原标签用于校准；本地提交 `ed9b2a8` push 后，gzs 成功 fast-forward，并在 `cloud-lite-pt210` 中执行锁定。
- 完整性：A/B 校准与共识均为 40/40、ID 集相同、无重复、无空标签、标签语法合法。SHA256：A `f547fdb08d2cd6379fcbc3a61399287919f093d342224d75962ce8482e55eeb2`，B `ff14ad31d411e27ac3150d3b8feadc3c808936aaca30fda05c48b6476aa71284`，共识 `8da009f62145dc4b0c36b3416902875d84f9618c3d7b03bbad26de80385b694d`，校准锁 `0ac14e4d9b384b19a1205cd6e78e050949c126d21884127beecb0c40622fb76f`。
- 止损线与真实状态：40 个校准单元已永久排除最终统计；状态为 `calibration_locked_awaiting_independent_reviews`。A/B 的两个 main CSV 均为 210 行但标签 0/210，仍是空白模板，故 `main_reviews_complete=false`、`human_results_available=false`；未计算 κ、AC1、IoU，未生成第三方分歧裁决，不得宣称61D2已完成。

### 2026-09-28：单 reviewer 主复核降级审计

- 目标：在人力不足、仅 reviewer A 完成主复核的现实约束下，保留人工观察，但禁止把单人判断伪装为独立一致性或金标准。
- 改动：新增单 reviewer 审计脚本，强制校验 210 个 main ID 与标签语法，结构性禁用 Cohen κ、Gwet AC1、reviewer IoU、第三方裁决共识、标注过程偏移门和“回到模型问题”门；packet 状态改为 `single_review_complete_insufficient_for_interrater_inference`。
- 人工限定：reviewer 明确说明“标注本身充满不确定性，黄框内可能同时含有多个语义，引用时需慎重，不可百分比相信”。该说明原文写入审计产物。CSV 中集合标签为 0、notes 为 0；仅 13 个 `boundary_mixed`、2 个 `uncertain`、9 个 `unobservable_nodata`，合计 24/210（11.43%）显式非硬标签。这只能视为被文件编码的不确定性下限，不能据此认定其余 186 项确定无歧义。
- 网络/读出：不训练、不推理；本地提交 `2aaf2ff` push 后由 gzs fast-forward，在 `cloud-lite-pt210` 中运行。reviewer A 文件 210/210 完整、无空值/重复/非法标签，SHA256 `9c00b9349862d7604884507955c64ddffe5eacab81983000016f2827eaf2fcda`；reviewer B 未完成且未作任何代填。
- 探索性描述：A 的 clear/cloud-shadow/thin-cloud/haze-cirrus/boundary-mixed/terrain-water-shadow/thick-cloud/uncertain/unobservable-nodata 计数为 41/66/37/27/13/9/6/2/9。原标签与单 reviewer 的 Thin/Shadow membership IoU 为 25.23%/42.06%（Thin 交27并107，Shadow 交53并126）；这些数值不是 reviewer 一致性、数据集错误率或共识标签性能，只可用于提出假设。
- 止损线与结论：`four_class_miou_may_be_primary_from_61d2=false`，原因是独立 reviewer 可重复性没有被测量。61D2 不能支持“标注过程偏移”或“回到模型问题”的核心结论；允许用途仅为带 reviewer 限定语的描述性、假设生成证据。单评审计 SHA256 `add477f6ede118c8221021e39b5d9ed4cb3c4d7672bb0f370c130df8980f907e`，packet summary SHA256 `5673d60f4ba0939e2e6ef4ab875abc39cb937fcffa9b87176bc9ccc15373d54d`。

### 2026-09-28：双 reviewer 主复核检查与待裁决状态

- 目标：导入 reviewer B 的尽力复核，恢复可计算的双人一致性端点，并只把真实分歧交给第三位专家。
- 改动：B 原始 CSV 原样封存，SHA256 `2f3855752e3d9c0217d2a66c830e184dadee1705d65b83249a252b8b87645afd`。其中 207/210 有显式标签；依据用户既有定义“空值表示纯色、无有效信息、无法评估”，在独立派生副本中仅将 3 个空值映射为 `unobservable_nodata`，不覆盖原件；派生 SHA256 `a4166d07bcb03237d956d4c661f88a0483b60d2d0b63de5aba2e6fcb43bca525`，provenance 保留三个 tile ID、输入/输出哈希与原因。
- 网络/读出：不训练、不推理；本地提交 `b8a4db4` push 后由 gzs fast-forward，在 `cloud-lite-pt210` 中通过 6 项测试并运行双评评分。A/B 均无集合标签和 notes，reviewer 关于多语义与低置信度的自述继续作为所有结果的强制限定。
- 总体结果：210 项 exact/compatible agreement 均为 62.38%，Cohen κ=0.5314，Gwet AC1=0.5820；Thin/Shadow reviewer membership IoU=36.73%/72.73%。全局 κ/AC1 均高于 0.40，Thin/Shadow IoU 均高于 0.30，故仅“总体 reviewer 可重复性最低门”通过；这不是四类主指标已获授权。
- 分层结果：旧错配分层160项的 κ/AC1=0.4828/0.5497，Thin/Shadow IoU=32.43%/72.73%；未见确认集50项为0.6635/0.6877、50.00%/72.73%。biome 差异明显，snow_ice 的 κ/AC1=0.2977/0.3297、Thin/Shadow IoU=0%/25%，其自身未过门；forest 的 Thin IoU=0% 但 union 仅1，不作稳定比例解释。
- 分歧与止损：79/210 项需裁决，其中原分层 boundary/interior 为44/35；分歧类型含 boundary 16、clear↔cloud-shadow 5、cloud-shadow↔terrain/water-shadow 5、thin↔shadow 3、thin↔thick 6、unobservable 8、uncertain 1、其他35。第三方模板已生成但仍为空白，状态固定为 `double_review_complete_awaiting_adjudication`，`formal_human_results_available=false`；原标签—裁决标签 IoU、标注过程偏移门和模型问题门仍不可计算。
- 产物：双评 metrics SHA256 `83be709e461469372952d085b670bdf01b423d31ae9d9dc1695d998414a13fc9`；裁决模板 SHA256 `a28c4201c8ede6ab8ea051b4ee17a5618adde231d275ad99bead61bea5372078`；packet summary SHA256 `55cc57a77ce007dc1c6ab1ea6cb625a4a9d7ddf0e944c707be3f09fff979db54`。

### 2026-09-28：第三方裁决交接包

- 目标：把第三位专家的工作量严格限制为 A/B 的 79 个真实分歧，不重复评审其余 131 个一致单元。
- 改动：生成只含79张对应11波段证据页、统一手册、中文裁决说明、空白 `final_label_set/rationale` 裁决CSV及成员哈希的专用ZIP；不含 sealed manifest、原标签、模型预测或方法身份。
- 操作口径：第三位专家可见 A/B 候选标签但不得按多数票，须依据相同证据和手册独立裁决；可使用最小二元素集合标签，`boundary_mixed/unobservable_nodata/uncertain` 必须单独使用；79行均须给出最终标签和简短理由，不得改写 tile ID 或 A/B 原判断。
- 止损线与状态：第三方文件返回并通过ID、原判断不可变、标签语法和理由完整性校验前，不运行最终硬门。当前仍为 `double_review_complete_awaiting_adjudication`，无新增科学结论。
- 产物：`phase61d2_adjudicator_packet.zip` 共83个成员、79张PNG、CRC通过、sealed成员0；SHA256 `1bcdd5848e5fb7c48a040e9d9c8a8cde50c9e55206171b2eda0dd68c189547bd`。

### 2026-09-28：待裁决阶段解释冻结

- 当前正式状态为`double_review_complete_awaiting_adjudication`，不是`phase61d2_complete`。A/B双评与总体最低reviewer门已完成，但79项真实分歧尚未裁决，故`formal_human_results_available=false`；原标签—裁决标签IoU、标注过程偏移门和模型问题门均不得提前填写。
- 总体κ/AC1与Thin/Shadow reviewer IoU越过0.40/0.30最低门，只说明四类硬标签主线未被最低门直接终止，不等于四类mIoU已自动获得唯一主指标资格。确认集κ高于旧错配分层集，说明模型错配条件抽样确实富集了困难位置，不能把旧分层集或总体210项比例外推为L8 Biome总体发生率。
- Thin reviewer IoU为0.3673，低于预注册“Thin和Shadow reviewer IoU均超过0.60”的强标注过程偏移门。因此在不修改事前规则的前提下，本Phase已不可能得到“强标注过程偏移”判定。第三方裁决仍有必要：若原标签—裁决标签的Thin与Shadow IoU均至少0.60，可按原规则回到模型问题；否则最终状态必须记为混合证据，不强行归因。
- 79项分歧中44项位于原分层的boundary区域，表示55.70%的分歧样本来自boundary stratum；这不等同于79项中有44项被reviewer标为`boundary_mixed`，实际分歧类型统计中的boundary为16项，二者不得混写。
- snow/ice分层κ/AC1为0.2977/0.3297、Thin/Shadow IoU为0%/25%，该分层自身明确未过门。不得为了提高总体一致性而事后删除snow/ice；最终报告必须保留全体结果并单列snow/ice失败，是否对该biome采用粗粒度、拒识或独立协议只能作为后续新假设另行预注册。
- 第三位专家只处理79项真实分歧，继续对原标签、模型预测、方法和抽样来源盲态，可见A/B判断与同一多光谱证据页，并必须记录裁决理由。裁决完成前不启动新模型、标签矩阵、辅助head或损失实验。

### 2026-09-28：第三方裁决完成与最终封账

- 目标：导入第三位专家对79项A/B真实分歧的最终标签，在不补写任何人工判断或理由的前提下执行预注册硬门；131项A/B一致样本直接保留其共同判断。
- 改动：原始裁决表完整封存为`completed_reviews/adjudicator_completed_raw.csv`；新增显式`--allow-missing-adjudication-rationale`降级开关，默认严格模式仍拒绝空理由。7项本地和7项远端协议测试均通过。本地提交`8e3c114`已push，gzs已fast-forward；不训练、不推理、不修改模型。
- 人工输入与降级：79/79项`final_label_set`合法且非空，计数为thin-cloud 24、cloud-shadow 15、thick-cloud 9、terrain/water-shadow 8、clear 6、haze/cirrus 6、unobservable 6、boundary 4、uncertain 1；集合标签0。`rationale`为0/79，故未满足交接包建议的解释完整性要求。本轮根据现实人力约束只接受标签，不生成、推断或代填理由；结果必须以“解释性降级、不能视为绝对真值”引用。裁决原表SHA256为`90fa08860ea01025b5c750db8b4506f65338e6c3f039017ba8e9b43e9231dce4`。
- 网络/产物：GitHub push与gzs pull成功；远端非交互shell需显式加载`/home/scv/miniconda3/etc/profile.d/conda.sh`后使用`cloud-lite-pt210`。预裁决metrics/summary按原哈希`83be709e...`/`55cc57a7...`保留为快照；最终metrics SHA256 `929a59fe212921879c87c08e7007404f37ca69f825c5a30f8b90ae5fc9338be9`，最终packet summary SHA256 `f2e7440dbf8259c5bcf21e013043271023c7ed06650f04e1906b3f759321cca7`。
- 真实结果：210项总体Cohen kappa=0.5314、Gwet AC1=0.5820，Thin/Shadow reviewer membership IoU=36.73%/72.73%。原标签与最终裁决标签的Thin/Shadow membership IoU=25.23%/40.46%。校准40项继续完全排除；最终状态为`complete_adjudicated`，packet标记`formal_human_results_available=true`，但必须同时携带三条人工不确定性与缺理由限定。
- 止损线与结论：kappa/AC1均不低于0.40，Thin/Shadow reviewer IoU均不低于0.30，因此`four_class_miou_may_be_primary=true`且不触发四类硬标签主线终止；但Thin reviewer IoU未超过0.60，不能支持强“标注过程偏移”，而原标签—裁决Thin/Shadow IoU也未同时达到0.60，不能归回纯模型问题。预注册结论为`inconclusive_mixed_label_and_model_evidence`。snow/ice分层失败及人工标注困难保持不变；不得把本结果解释为原标签错误率、可靠金标准或对全部L8 Biome的发生率估计。


---

<a id="phase-62"></a>

## Phase 62 实验记录

### 2026-09-25：部分标签建模审计启动

- 目标：验证 `Shadows?=no` 图像中 raw ID0 采用集合标签 `{clear, shadow}` 是否优于直接 ignore，同时保留 thin/thick 普通监督。
- 改动：不增加网络参数或结构模块；三组共用 Phase52 冻结 Mask2Former head 与既有 380,577 参数 target adapter、同一初始化、优化器、增强、4000 steps 和 seed 62/63/64。仅改变 53 张 `Shadows?=no` 图像 raw ID0 的数据/损失口径。
- 对照：`complete12`；`ignore65`（12 complete + 53 thin/thick，ID0 ignore）；`partial65`（12 complete + 53 thin/thick，ID0 使用 `-log(p_clear+p_shadow)`）。
- 评估：target-val 仅使用 `Shadows?=yes` 的 963 张、8 scenes；source-val 535 张按既定 disabled-target 路径检查遗忘；target-test 与 CloudSEN internal-test 继续封存。
- 止损线：相对 `ignore65`，partial 组 Thin/Shadow 调和均值必须提高，Thin 与 Shadow 的三 seed 均值都不得下降；分层 paired scene bootstrap 95% CI 下界必须大于 0 且至少 6/8 scenes 同向；三个 partial seed 的 source mIoU 均须 ≥73.93；non-clear mIoU 必须提高。任一失败即停止复杂标签转移矩阵路线。
- 结果：尚未产生，严禁提前填写。

### 2026-09-25：预检通过并启动 3×3 seed

- 目标：在训练前验证冻结选择集、USGS 状态、三组像元映射、部分标签公式、参数公平性与梯度可达性。
- 改动：新增无参数 `PartialLabelFrozenHeadEncoderDecoder`；常规监督保持原生 Mask2Former loss，只有保留值 254 的像元追加 `-log(p_clear+p_shadow)`。预检首次发现有效 head 为 Mask2Former，改用仓库 Phase53 同口径的 dense semantic prediction；另修复两处仅影响审计器的 MMEngine lazy/serialized dataset 读取问题。所有修复均在训练启动前完成。
- 网络/运行：本地代码经提交 `b57fe10`、`90fc20c`、`89120aa`、`0afaca7` push；gzs 多次遇到 GitHub GnuTLS 非正常终止，最终以单次 HTTP/1.1 pull 成功同步。远端环境 `cloud-lite-pt210`，串行训练主 PID `449138`，避免九轮并发争用 GPU。
- 数据：manifest SHA256 `885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b`；选择集 SHA256 `564ad4e2c27d2948c725cf4c5c482159888c2324fe0ab4bdeee25fe774cebc55`；12 张 complete 与 53 张 `Shadows?=no` 精确复现。53 张 no 图像含 raw ID0/2/3 像元 7,051,817/2,295,161/4,546,654。
- 止损线：训练前必须满足组规模 12/65/65、raw ID0 映射 0/255/254、无新增可训练参数、解析损失为 `log 2`、真实模型反向梯度有限且可达原 adapter；任一失败不得训练。
- 结果：全部预检门通过。三组模型配置完全相同，可训练参数均为既有 380,577；均匀四类概率下 `{clear,shadow}` 部分损失为 0.69314718；真实 smoke partial loss 0.43044755，21 个既有可训练张量获得有限非零梯度。已启动 `complete12/seed62`，100/4000 step 时总损失 20.8223、partial loss 0（符合该组无部分标签像元），训练与最终指标尚未完成。

### 2026-09-26：3×3 seed 完成，部分标签继续门失败

- 目标：比较 `complete12`、`ignore65`、`partial65` 的三 seed target-val、source-val 与 scene bootstrap，决定是否发展复杂标签转移矩阵。
- 改动：无新增模型或训练改动；九轮均完成 4000 steps，随后逐 checkpoint 评估 963 张 `Shadows?=yes` target-val 与 535 张 source-val，并执行 10,000 次 seed×scene 分层配对 bootstrap。target-test 与 CloudSEN internal-test 未读取。
- 网络/运行：远端串行运行于 `cloud-lite-pt210`，完成时间 `2026-09-25T23:57:02+08:00`；九个 `TRAIN_COMPLETE`、总完成标记和九份 evaluation 均存在。训练/评估日志未发现 traceback、OOM、NaN 或 Inf；partial 组三轮 `partial.loss_set_nll` 均为有限非零值。
- 结果（mIoU/Thin/Shadow/弱类调和均值/non-clear mIoU，三 seed 均值）：`complete12`=38.905/25.306/11.261/15.563/26.423；`ignore65`=42.416/33.320/3.189/5.786/32.764；`partial65`=42.437/30.774/1.966/3.673/30.916。既定 disabled-target source mIoU 三组各 seed 均为 73.980，遗忘门通过。
- 相对 `ignore65`：partial 的 mIoU +0.021，但 Thin -2.546、Shadow -1.222、调和均值 -2.113、non-clear mIoU -1.848；Clear 单独 +5.629。三个配对 seed 的调和均值差为 -6.057/-0.580/+0.298，明显不是弱类一致改善，而是近乎完全由 clear 类抵消。
- scene bootstrap：弱类调和均值配对差均值 -2.091，95% CI [-6.132, 0.177]，改善概率 0.0695；8/8 scenes 的三 seed 平均差均为负，稳定方向与继续门相反。
- 止损线：调和均值提高、Thin 不降、Shadow 不降、bootstrap CI 正向、至少 6/8 scenes 正向、non-clear 提升和“非 clear-only”七项均失败；只有 source 遗忘、运行完整性与封存协议通过。
- 判定：`stop_complex_label_transfer_matrix`。当前 `{clear,shadow}` 部分标签基线无效，不发展复杂标签转移矩阵；不得用 +0.021 mIoU 宣称改善。
- 产物：`preflight.json` SHA256 `d56086bfe686e836a3f7d7c06b8b96f6dd3d7842726ddd827d60684033098036`；`summary.json` SHA256 `2244e5edc6d3db55f763997a60545b6325f0730a8fc4e8959562ca5d1284af45`；`evaluation_console.log` SHA256 `e9725b0207ca86731d0f9fb0f0cf526fce20c66e7d435090a78ac16171d37e1b`；`phase62_console.log` SHA256 `205cba49178074380a02010a6e3cc7a3df16ffc22f63812bf6a910da3a27b07a`。最终结构、样本数、运行数和决策断言复核通过。

### 2026-09-26：机制解释与路线封存

- 对允许集合`S={clear,shadow}`使用`-log(p_clear+p_shadow)`时，损失只要求概率质量进入该集合，不提供区分clear与shadow的额外监督。在当前初始化和类别先验下，模型可通过提高占优的clear并压低集合外类别来满足目标；观测到Clear `+5.629`而Thin/Shadow同时下降，与这一不可辨识退化一致。
- scene bootstrap 95% CI `[-6.132,0.177]`包含0，因此不得宣称partial相对ignore的平均下降达到常规双侧95%统计显著；但8/8 scenes方向为负，且Thin、Shadow、弱类调和均值、non-clear mIoU和预注册的七项继续门同时失败，足以支持预注册止损。
- source mIoU保持73.980只说明关闭目标路径和源域保持有效，不构成目标域方法有效性。总mIoU `+0.021`由Clear改善抵消弱类下降产生，严禁表述为性能提升。
- 精确结论是“当前`{clear,shadow}`集合监督在当前数据、初始化和损失下无效”，不能扩大为“所有部分标签学习均无效”。然而同一弱监督不含clear/shadow内部可辨识信息，不再授权通过调权重、温度、类别平衡或更高自由度标签转移矩阵继续相邻试错。
- 正式执行`stop_partial_label_and_transition_matrix_mainline`。下一步若继续，只允许先取得经校准、可重复的人工复核证据，再另行预注册与旧四类强制叶分类不同的新任务定义；在此之前不启动新的标签矩阵、辅助head或损失变体。


---

<a id="phase-63"></a>

## Phase 63 实验记录

### 2026-09-29：选择性多粒度可行性协议、工具链与scene冻结预检

- 目标：不训练新分割网络，先检验“现有模型能否识别哪些中心像元具备四类细分条件、哪些只能退回粗粒度或拒识”。冻结Phase61D2的210个主复核点、79条缺失理由、snow/ice分层和Phase55模型对；160项仅用于风险开发，50项仅允许最终确认。
- 改动：新增Phase63预注册、scene锁与泄漏审计、现有checkpoint不可靠性证据导出、逻辑回归/固定GAM/isotonic风险评分、scene bootstrap与风险—覆盖曲线；新增无参数层级概率/router原型，不注册训练配置、不含深层risk MLP。四类模型输出顺序固定为`clear/thick/thin/shadow`；单四类softmax的层级恒等式明确标为不可用，只有独立coarse概率才能产生层级冲突特征。210项均按黄色角标中心像元评价，禁止称为dense mIoU。
- 数据冻结：sealed manifest SHA256 `7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77`；source-only Phase22 seed42 iter40000 checkpoint SHA256 `d33a81337544e294a42007c6b353ee13c901ebb1e1f5261fd1d95a79e3888012`；adapted Phase52A-fix seed52 iter1000 SHA256 `9655ee83c63840ecb31351efef0582676529c318fd0d1eca5fd0929db702eb2e`。最终210项中明确四类162项（dev126、confirmation36）；其余48项不强制映射到四类，而作为细粒度不可判/失败保留。79条裁决理由保持0/79，未补写。
- 网络/运行：只通过ssh gzs在授权仓库内只读核验已有checkpoint并同步真实sealed manifest；同步代码时两次SSH连接在密钥交换阶段被远端关闭，第三次恢复后正常fast-forward。未读取新数据、未下载第二域、未启动GPU推理或训练。本地与远端`cloud-lite-pt210`各20项协议/原型测试及语法检查通过；远端复现同一scene审计哈希且未生成split lock。用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`及`outputs/`改动未覆盖。
- scene真实结果：现有development/confirmation都来自同一8个scene，重叠8/8；各scene的dev/confirm数为14/8、14/8、14/8、21/8、32/6、26/4、32/4、7/4。snow/ice 44项原样保留（28/16）。审计按规则fail closed，`phase63_split_lock.json`未生成；审计产物SHA256 `cbeb56b9b2964110bb5162e23e78127f9ff1469162f1c31e536784ebc9b11ee2`。
- 止损线：在合法scene lock出现前，禁止导出带confirmation标签的证据、拟合风险模型或计算63C AUROC/coverage/bootstrap。因此本轮没有AUROC、CI、覆盖率或生死门结果，严禁填报。保持旧50项角色不变与scene零泄漏在当前210项上不可同时满足；优先方案是从此前未见scene取得新的确认集，备选是明确废止旧160/50角色后按完整scene重划，但旧50项不能继续称为未触碰确认集。
- C组状态：层级合并与外部风险router原型为无参数模块，fine/coarse/abstain、非有限风险拒识和独立coarse冲突测试通过；没有训练或性能结果，不能称为方法有效。
- D组状态：资料审计推荐Deep Fmask Sentinel-2雪冰/高山域作为第二标注域；它有45个scene和原生clear-land/cloud/shadow/snow/water标签，可做三类/二类与snow压力测试，但没有thin/thick像元真值、与CloudSEN同为Sentinel-2，不能称第二传感器或四类验证。Deep F-Mask Stage-4仅是论文锚点，未下载数据、未复现；报告SHA256 `8257076fb35f5f313b02c04374a0f19d36ab9b71d28890445ee73d4e631d06d9`。
- 当前判定：`blocked_before_phase63c_by_scene_leakage`。这不是选择性多粒度命题已经失败，也不是已经通过；只是现有确认设计不能给出无泄漏判定。不得以新模型或注意力模块绕过该数据门。

### 2026-09-29：新增scene独立人工确认包（等待复核，未运行63C）

- 目标：在任何风险模型拟合前修复旧160/50的scene泄漏。旧50项不再承担Phase63最终确认角色；新建两个预注册队列：32项严格`target_val`确认队列，以及18项Phase50零选中patch的snow/ice训练候选scene压力队列。后者必须单列，不能伪称严格未见验证集。
- 改动：新增`prepare_phase63_human_confirmation.py`及3项抽样/防泄漏测试，并让既有61D2图板渲染器接收阶段标题而不改变默认行为。抽样是像素标签盲和模型盲，但明确使用`Shadows?`及biome协议元数据分层；patch和中心坐标均由冻结输入与固定salt确定。预注册与代码提交为`af94a04`。
- 数据与结果：确认集共50项、11个scene；32项来自8个此前未进入旧160项、也未进入Phase50选中训练scene的`target_val Shadows?=no` scene，18项来自3个Phase50零选中patch的`target_train snow_ice` scene。与旧development scene重叠0，与Phase50选中训练scene重叠0，`target_test`读取0。reviewer包含40张旧校准图和50张新确认图，共90张1040×1210 RGB图；所有新图展示B1–B11、推荐合成、太阳方位和必要元数据。
- 盲态核验：ZIP共98个成员；5份待填CSV分别为40/40/50/40/50行，除`tile_id`外全部为空；包内无sealed manifest、原标签、模型输出或风险分数。reviewer ZIP SHA256 `dc16956f960c4d7e86329d41b06ddd98b35ce26311cc1359b3965fd8cdafab85`；sealed manifest SHA256 `59f64b9063c37c2de367481dc8b3cc3be08c3ad8df50d5a031338650545fb3e1`；packet summary SHA256 `24ef62f883b3e66807bf3c5fe7be778f641ecfd06499c7b60c96e2c0fafcb828`。抽查snow/ice及非snow图板，波段、合成、中心标记和方位图可读。
- 网络/运行：本地30项相关测试通过；代码push后，gzs授权仓库fast-forward到`af94a04`，远端`cloud-lite-pt210`中22项工具协议测试和8项无参数router测试通过。只读访问已授权的`/home/scv/shared/data/l8_biome_raw`生成图板，写入仅发生在远端仓库及本地仓库的Phase63 work dir；通过SSH/SCP同步成功。未启动GPU推理或训练，未改写用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`或`outputs/`。
- 止损线：当前状态为`awaiting_new_reviewer_calibration`，不是63C通过。A/B必须先独立完成40项校准并讨论记录共识，再独立完成50项确认；第三专家只裁决真实分歧。两份原始判断、裁决和哈希冻结前，禁止查看确认标签拟合风险、计算AUROC/coverage/bootstrap或启动新网络。最终必须分开报告32项严格确认队列与18项snow/ice压力队列；不得用合并50项冒充单一总体的无偏发生率。

### 2026-10-03：校准读出与评价单位缺陷修正

- 目标：核验新Phase63校准是否完成，并在继续confirmation前检查低一致率是否混入界面定位歧义。标注程序未覆盖根目录模板，真实完成记录位于`.annotation_history`最新快照；A、B与consensus各40行，其中同一第40项三方均为空，按用户既定规则记为纯色/无有效信息，39项可评价。
- 真实校准读出：A/B exact及compatible agreement均为9/39（23.08%）；Cohen κ `0.1308`，Gwet AC1 `0.1253`；Thin membership IoU `16.67%`，cloud-shadow `41.67%`，thick `0%`，clear `14.29%`。30项分歧中13项涉及`boundary_mixed`、4项为cloud-shadow↔terrain/water-shadow、2项thin↔thick、1项thin↔cloud-shadow。A使用`boundary_mixed` 1次，B使用12次。讨论后consensus与A一致28/39、与B一致17/39，不能当作独立一致性证据。A/B/consensus快照SHA256依次为`09ced51a68c40a342264132b7dce2ab58f40ac892b73ebdfe143c856a7ca47e2`、`1d1b43e21893cd9ae4889c5ffb2a52cc69272bfa9f2f6880407ae572ac789ff3`、`e072d0fd5c09f31ada811c3285b10ab7dd5fd4ef00b731db119be674e057194c`。
- 关键修正：旧图中文字虽称待判像元，但黄色角标在源分辨率上实际围住约20×20像元，不能唯一指定评价单位。因此上述低κ同时受目标定位歧义和语义歧义影响，不能全部归因于细粒度标签不可复现；Phase61D2旧数字不删除，但后续引用必须附此限制。
- 改动：新增16项精确像元UI pilot生成与评分工具、4项测试及独立手册。页面分离192×192上下文与21×21最近邻像元视图；空心框只包围一个像元且不遮挡内部。先填目标可定位性，再填可辨识性；只有可细分时才填语义类别，并记录耗时。pilot按8个biome各2项抽取此前未人工复核、未被Phase50选中的target-train patch，排除所有既有复核scene和target-test；只用于UI，不进入模型或确认统计。
- 网络/运行：本轮截至记录时仅完成本地语法检查及26项Phase工具测试，未访问远端原始数据、未生成pilot图、未启动GPU或训练。用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`与`outputs/`未改动。
- 止损线：正式confirmation继续暂停。UI pilot须由两人独立完成且每图只看一次；双方可定位率均≥95%、可辨识性exact≥70%且AC1≥0.40、双方均判可细分至少6项且其语义exact≥50%才允许重做正式confirmation。新旧项目难度不同，前后差异不得宣称为UI改动的因果效果；pilot尚无人工结果或Phase63生死门结果。

### 2026-10-03：精确像元 UI pilot 包生成完成（等待独立复核）

- 目标：以最小工作量检验“唯一中心像元 + 上下文/局部双视图 + 可辨识性先行”能否消除目标定位歧义；不训练、不评价模型、不读取confirmation标签。
- 改动与产物：代码提交`c643977`在gzs fast-forward后，以`cloud-lite-pt210`和只读原始波段生成16项pilot。8个biome各2项、8个此前未人工复核的target-train scene；与既有复核scene重叠0、与Phase50实际选中patch重叠0、target-test读取0。每页含192×192上下文、4组21×21合成和11个21×21单波段最近邻视图，显示精确`(x,y)`；空心框只包围一个源像元且中心内部可见。
- 核验结果：reviewer ZIP共23个成员、16张1040×1210 RGB图、A/B各16行空白表；包内无sealed manifest、原标签、模型预测或risk score。目视抽查普通biome与snow/ice页面，像元格、中心框、十字线、波段和太阳方向可读。ZIP SHA256 `3ea92af06157c405529e202768ce1b4e83919312dcd5f22f5b73ceb9912ff568`；sealed manifest SHA256 `7503977e55cd35be3c30c3c376efd2f51b09907ebd95df450bfa88da07153627`；packet summary SHA256 `bbbe6e3d47fd124f864f9831f13a6553af80a4afff165bf7338417de80ef1298`。
- 网络/运行：首次SSH连接在密钥交换阶段被远端关闭，重试后正常；远端4项pilot测试通过并完成生成，SCP同步成功。只读访问`/home/scv/shared/data/l8_biome_raw`，写入仅发生在授权仓库；无GPU推理或训练。用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`和`outputs/`未覆盖。
- 止损线与当前结果：状态为`awaiting_independent_review`。当前只有UI与盲态核验结果，没有人工一致率、耗时、模型AUROC或Phase63生死门结果。A/B必须独立、每图仅一轮并在原始CSV冻结前不讨论；pilot通过预注册UI门后才允许重做正式confirmation。

### 2026-10-04：semantic-coverage exact-pixel UI pilot 重新预注册并生成

- 目标：废止无语义哈希生成的16项UI pilot，暂停原随机中心50项confirmation；以原标签仅作sealed抽样层，生成20项只检验UI与标注流程的精确像元pilot。固定配额为clear/thin/thick/cloud-shadow interior各4项及thin↔thick、cloud↔shadow、clear↔shadow边界2/1/1项；优先不同scene、每scene最多2点、每patch最多1点、每个interior类至少3个scene。
- 改动：新增20项生成、冻结、评分与测试工具及操作手册；中心像元必须距任意标签边界`>3 px`，在不降低scene覆盖等硬约束时确定性优先`>=5 px`；边界点来自预注册类别对的真实界面。可观测性门固定为目标B2/B3/B4有限非零且居中192×192上下文RGB有效比例至少50%。纠正原始L8 fixedmask映射为`0=Fill,64=shadow,128=clear,192=thin,255=thick`，不得与转换后的四类ID0混淆。
- QA过程：首个候选包因把原始Fill=0误当clear而出现3/20张纯黑页，已标记`qa_rejected_before_review`并隔离；第二包虽满足`>3 px`硬门，但在`>=5 px`偏好尚未成为确定性审计字段时即被`qa_superseded_before_review`隔离。两包均未交给reviewer、不得评分。旧16项标为`superseded_before_review`；旧50项保持未读取并标为`suspended_not_formal_confirmation`。
- 真实结果：最终20项覆盖16个不同scene，最多2点/scene、1点/patch；四个interior名义类各覆盖4个scene，边界配额严格为2/1/1；16个interior中15个距离`>=5 px`，其余1个距离`3.606 px`，仍满足`>3 px`硬门；边界点最大类别对距离0 px。最小RGB有效上下文比例`0.6106`；20页均为1040×1210 RGB，无低信息/纯色页，最少上下文颜色数66。target-test读取0、模型数组读取0、与既有复核scene/patch重叠0。
- 产物核验：A/B CSV各20行且除ID外全空，ID与20页完全一致；ZIP共27个成员，内部SHA清单全匹配。最终reviewer ZIP SHA256 `bdae9a616e40f894b7e780672d70467e72e94b456094bd7bb8b5b2d76446b74a`；sealed manifest SHA256 `38e1168cf3a34636debdccb89d2b1b1e54b0a52d1d58cc0020937e82f0678fd6`；packet summary SHA256 `21cde11833697ec187a600b4712462fa49158c8ed04257a30d9bfa4ee77fea7`。
- 网络/运行：本地修改均先测试、commit和push，gzs授权仓库逐次fast-forward；远端固定使用`cloud-lite-pt210`，最终8项工具测试通过。只读访问授权的`/home/scv/shared/data/l8_biome_raw`，写入仅发生在本地/远端仓库Phase63 work dir；未训练、未启动GPU推理，未覆盖用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`或`outputs/`改动。
- 止损线：本20项只能检验UI与流程。A/B必须独立完成；两人可定位率各至少19/20、零坐标/页面/格式错误、可辨识性exact至少70%、interior中双方均判可细分至少10项且其语义exact至少60%才通过。只报告名义层原始计数；A/B原始判断冻结前禁止解封sampling字段。当前状态`awaiting_independent_review`，没有人工一致率、类别准确率、模型AUROC、coverage或Phase63C生死门结果。

### 2026-10-04：semantic-coverage UI pilot 双评冻结与评分

- 目标与冻结：A/B各完成20项。格式校验发现A最初有3项`target_locatable=no`但仍填`mixed_boundary`，由reviewer自行修正为`no`且后续判断留空后再次校验通过；程序未替改原始判断。nominal层解封前冻结A/B SHA256为`0ed3a23d7f93d3f1d4e102cb221153aaa8c0ddd418a0ad7a7e199b46e92d73bb`/`e4a165894e36e46e7aac0abd6f96eb09c86f3373589893143436b76b3235a63f`，lock SHA256为`6b43b587e33fbc46b347713a4178984b11f8e681c4b3849d74bce88a27ca862f`。
- 真实结果：A/B可定位分别为17/20与20/20，共同可定位17项。可辨识性exact为15/17（88.24%），Cohen κ `-0.0625`、Gwet AC1 `0.8677`；负κ来自几乎全部选择`fine_classifiable`的极端边际分布，不能脱离exact/AC1解释。interior中双方均判可细分15项，语义exact为11/15（73.33%），κ `0.6429`、AC1 `0.6467`。平均耗时A/B为14.28/28.92秒。
- 名义层原始计数：A在clear/shadow/thick/thin四个interior层的可细分数为4/3/4/4，另1个shadow为证据不足；B四层均为4。A的语义计数依次为`clear2+haze1+shadow1`、`shadow3`、`thick4`、`haze1+clear1+thick2`；B依次为`clear4`、`shadow3+terrain/water-shadow1`、`thick4`、`haze3+clear1`。A有3/4个名义boundary项不可定位，B的4个boundary项为2个可细分、2个证据不足。reviewer–nominal-label concordance为A 9/15、B 11/16，仅是描述性一致，不是人工准确率。
- 止损与判定：六个预注册门中五个通过；唯一失败项为A可定位17/20低于19/20，因此`ui_pilot_pass=false`。本轮只说明当前UI/字段解释流程尚未稳定通过，不证明四类标签不可复现，也不评价模型。不得据此启动正式confirmation或Phase63C风险模型；metrics SHA256为`f163c36e903a2a6286682da4b31b936a32a08b2c0d3268aacdc22099b479e61e`。


---

<a id="phase-64"></a>

## LOG64 — 异构标签空间的层级一致性参数高效域适配

### 2026-10-04 — Day 1 数据与协议冻结

- 目标：停止逐像素人工裁决路线，先核验是否能用公开数据构造 `surface_visible/cloud/shadow` 共享父类与数据集特定原生子类，不新增人工标签、不训练新模型。
- 改动：新增 Phase 64 预注册、四数据域机器可读 registry、无第三方依赖的协议审计及 4 个回归测试。父类 `Clear` 修订为语义明确的 `surface_visible`；snow/water/flooded 保留原生子类，Fill/Unlabelled 显式 ignore；L8 完整三父类监督固定只用 `Shadows?=yes`。
- 网络情况：本地首次下载因 Windows Schannel 凭据失败；获准后从 PANGAEA 官方地址成功取得 2.1 MB Deep Fmask 标签包。SHA256 为 `eb53c2dd795c1541afcef4f8cf5398953748883c82a79d9512dfa3fbb28e7bcd`；实测类别 ID 为 0–5，SAFE 清单为 validation 22、test 23。未下载 Sentinel-2 SAFE 原始影像。
- 止损线：Day 2 前必须有两个目标域 loader、数据 hash 和 scene 泄漏审计；Day 3 要求两个目标域同向、父类 mIoU 平均至少 `+2.0`、source 下降不超过 `1.0`、MsRE 新增参数不超过 `0.50M`、Shadow 不灾难性坍缩。未达即停止当前组合，不新增模块。
- 真实结果：registry 审计通过，4 个单元测试通过；公开候选为 CloudSEN12 High、L8 Biome、Deep Fmask、SPARCS。当前只有 L8 一个目标域 loader 已实现，`day2_two_target_execution_ready=false`；因此 Day 1 科学协议通过、两目标训练执行门未通过，本轮没有模型指标。
- 同步核验：提交 `be32d0c` 已 push 并由 gzs fast-forward pull；远端 `cloud-lite-pt210` 中 4 个测试与 registry 审计再次通过。只检查授权路径后确认 CloudSEN 与 L8 derived 数据目录存在，获准只读的 `/home/scv/shared/data/l8_biome_raw` 也存在；未访问其他共享数据目录。

### 2026-10-04 — Day 2 SPARCS 准备与路径边界事件

- 目标：把 USGS SPARCS 作为第二个可执行目标域，冻结 80 scene 的项目划分，并建立父类/native label loader；不训练模型。
- 改动：新增安全 ZIP/CRC、80 mask、shape、label、source-scene 与 60/10/10 scene-disjoint 审计；新增 manifest loader；新增按结构严格满足 `P(parent)=ΣP(native child)` 的共享父类/数据集条件子类 head 原型。远端与本地的协议、loader、概率恒等式和梯度测试均通过。
- 网络与真实观测：USGS 官方归档下载完成，大小 `1,555,733,976` bytes，SHA256 `5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a`。首次审计读到 400 个成员、80 个独立 Landsat scene、80 个 1000×1000 mask、标签仅 0–6；固定哈希划分为 60/10/10 scene，三个父类在每个 split 均有像元。target-test 的 native `flooded` 为 0 像元，后续不得为该 split 报告 flooded IoU。
- 权限事件：随后发现 `/home/scv/Cloud-Adapter-light/data -> ../shared/data/`，所以归档与解压目录实际解析为未授权的 `/home/scv/shared/data/sparcs`。发现后立即停止对该目录的读取、移动和删除；首次审计结果只作故障记录，不作为正式执行就绪证据。代码已增加 `allowed_root` 的 symlink-resolved containment 硬门，并把默认目录改到仓库实体目录 `phase64_data/sparcs`。
- 当前止损：在用户明确授权如何处置误落盘目录前，不再访问 `/home/scv/shared/data/sparcs`；正式 SPARCS 审计必须从官方 URL 重新下载到解析后仍位于 `/home/scv/Cloud-Adapter-light` 的路径并复跑。没有正式第二目标域前不得启动 Day 2 双域训练。

### 2026-10-04 — Day 2 基线实现与训练前硬门

- 目标：在不读取越界数据、不启动正式训练的前提下，把共享三父类 readout、full fine-tuning、标准 LoRA、MsRE 与 head-only 基线实现到可审计状态。
- 改动：Day 2 快速筛选固定为 RGB，避免把标签层级收益与六波段输入变化混杂；公共六波段移至过门后的 Phase 64B。新增 CloudSEN/L8 父类 dataset view、SPARCS loader、标准 DINO LoRA、显式 full-FT backbone、父类 checkpoint 转换器、2 target × 4 method 配置矩阵及权重/参数审计。转换器只移除 `decode_head.cls_embed.{weight,bias}` 两个四类不兼容张量，其余 379 个张量保留。
- 网络情况：USGS 官方 SPARCS 重下载在仓库实体目录续传到 `829,991,528 / 1,555,733,976` bytes 后，服务端连续返回 HTTP 500 或 TLS EOF；官方页面未发现第二下载镜像。保留部分文件等待同 URL 续传，未读取或利用共享区误落副本。
- 止损线：正式训练前必须取得 CloudSEN derived 与 L8 derived 的明确只读授权；SPARCS 必须在仓库实体目录完成 SHA256/CRC/scene 审计；配置、checkpoint 和可训练参数硬门必须全部通过。任一额外缺失/意外权重、MsRE 超过 `0.50M` 或数据路径越界均停止。
- 真实结果：配置矩阵 `8/8` 通过。Phase 22 输入 checkpoint SHA256 为 `d33a81337544e294a42007c6b353ee13c901ebb1e1f5261fd1d95a79e3888012`，转换输出 SHA256 为 `e518be667397ebb5a9f32781436bca6c2e0f22aa0c1ad7e6de9ad6448ac813b2`。模型实例化/加载审计通过；LoRA 可训练参数 `294,912`，MsRE `380,577`，二者训练名称与声明完全一致，MsRE 低于硬上限。尚未训练，因此没有 mIoU、Shadow IoU 或 source forgetting 结果。
- 权限阻塞：仓库 `data` 实际解析到 `/home/scv/shared/data`；CloudSEN 根为 `/home/scv/shared/data/cloudsen12_high_l1c`，L8 manifest 指向 `data/l8_biome`（实际为 `/home/scv/shared/data/l8_biome`）。此前只授权 `/home/scv/shared/data/l8_biome_raw`，故未读取这两个 derived 目录，训练保持暂停。

### 2026-10-04 — Day 2 数据解封、源基线与首个目标作业

- 目标：在新增的 `/home/scv/shared/` 访问授权下完成第二目标域硬审计、冻结 source-only 父类基线，并按一次一个作业启动目标域快速筛选；共享目录保持只读。
- 改动：从共享区已有完整 SPARCS 归档只读复制到仓库实体目录，修正 `*_qmask.tif` 与 `*_data.tif` 的精确配对规则；新增数据就绪审计、受保护训练 launcher、source retention 配置和显式父类 IoU metric。独立评估器改用 Pillow 保留 CloudSEN 调色板 PNG 的类别索引；OpenCV 会把该索引错误展开为三通道颜色。远端调色板回归测试 `2/2` 通过。
- 网络情况：未再次联网下载数据。只读来源 `/home/scv/shared/data/sparcs/l8cloudmasks.zip` 与仓库副本均为 `1,555,733,976` bytes，SHA256 均为 `5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a`；失败的部分下载文件独立保留，未覆盖或删除。
- 止损线：两个目标域相对 source-only 必须同向，父类 mIoU 平均提升至少 `+2.0`，source mIoU 下降不超过 `1.0`，MsRE 新增参数不超过 `0.50M`，Shadow 不得灾难性坍缩；target-test 继续封存。未满足即停止当前数据/骨干组合，不新增模块。
- 真实结果：SPARCS 正式审计通过：400 个归档成员、80 个样本/scene、60/10/10 split、mask 均为 1000×1000 且 ID 仅 0–6，manifest SHA256 为 `8b4305280331e42853e0cca726d54a1ca27feddc6f9e6d6c4fdfb10079353b99`。数据就绪审计通过：CloudSEN 8490/535/975、L8 `Shadows?=yes` 1560/963、SPARCS 60/10，路径、父类映射和三父类覆盖均合法。三类 source parent 训练 4000 step 完成，source-val mIoU `77.82`（surface/cloud/shadow `88.30/83.84/61.31`），checkpoint SHA256 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。
- source-only 冻结基线：CloudSEN source-val mIoU `77.85`、L8 target-val `58.34`、SPARCS target-val `57.71`；对应 Shadow IoU 为 `61.38/27.15/22.03`。评估张数为 `535/963/10`，L8 为 8 个 scene、SPARCS 为 10 个 scene，均完成 2000 次分组 bootstrap；没有读取 target-test。CloudSEN 缺少可恢复的 scene ID，因此其 535 单位 bootstrap 实际是 patch 级，不能表述为独立 scene 证据。
- 当前状态：`L8/shared_parent/seed64` 已由 source parent checkpoint 启动；iter 50 总 loss `15.2145`，各项有限，进程存活。CUDA/CuBLAS、grid sampler 和 cumsum 在 deterministic warn-only 模式下报告不可完全确定性，必须作为复现实验限制保留，不能宣称 bitwise deterministic。30 分钟心跳负责核对并串行推进矩阵。

### 2026-10-05 — Day 3 单 seed 快速筛选与止损判定

- 目标：完成 RGB、seed 64、4000 step 的 `L8/SPARCS × shared_parent/full/LoRA/MsRE` 冻结矩阵；每个真实最佳 target checkpoint 在同一 CloudSEN-val 535 张上复核 source forgetting，target-test 始终封存。
- 改动：新增受保护的 source-retention launcher 与最终汇总/止损脚本；所有训练、评估均有 PID、exit 0 与完成标记。独立评估器的 CloudSEN 调色板 PNG 读取错误已用 Pillow 索引读取修复，远端回归测试通过。没有修改数据、标签、冻结超参或已有 Phase 60–62 记录。
- 网络情况：远端 SSH 曾两次瞬时返回 `kex_exchange_identification: Connection closed by remote host`，重试后恢复；训练中 step 时间有波动，但项目内始终只有一个训练主进程，未发现 OOM、Traceback、NaN/Inf。共享目录仅只读使用，未修改或删除。
- 止损线：核心 MsRE 必须在两个目标域相对 source-only 同向提升、平均父类 mIoU 增益至少 `+2.0`、每域 source 下降不超过 `1.0`、参数不超过 `0.50M`；Shadow 灾难性坍缩预先操作化为任一目标 Shadow `<10` 或相对基线下降超过 `10` 点。所有日志 loss 必须有限，target-test 必须封存。
- L8 真实最佳结果（source-only `58.34`, Shadow `27.15`）：shared-parent `64.10/35.63`，source `63.21`；full `51.51/13.83`，source `38.21`；LoRA `62.80/35.71`，source `66.61`；MsRE `61.03/26.79`，source `77.81`。前三者分别 source 下降 `14.64/39.64/11.24`；MsRE 仅下降 `0.04`，目标 mIoU `+2.69`，Shadow `-0.36`。
- SPARCS 真实最佳结果（source-only `57.71`, Shadow `22.03`）：shared-parent `69.65/42.98`，source `69.92`；full `61.05/34.56`，source `50.54`；LoRA `68.82/43.82`，source `68.16`；MsRE `67.15/41.60`，source `77.81`。前三者分别 source 下降 `7.93/27.31/9.69`；MsRE 仅下降 `0.04`，目标 mIoU `+9.44`，Shadow `+19.57`。
- 判定：MsRE 两域同向，平均目标增益 `+6.07`，source 每域下降 `0.04`，可训练参数 `380,577`，Shadow 非灾难性坍缩；完整矩阵 loss 有限且 target-test 封存，所有预注册 Day 3 硬门通过，决策为 `continue_to_phase64b_common_six_band`。但这只是单 seed 点估计快速筛选，适配模型尚未做 scene bootstrap，CUDA warn-only 也非 bitwise deterministic；不得把它写成稳健性或统计显著性结论。
- 产物：最终机器可读汇总位于 `src/work_dirs/phase64_final_summary/summary.json`；MsRE 最佳 checkpoint SHA256 为 L8 `fcfa934e37e1ed1569e4eded2547ee999626840a6e03e9a58395ab6d5e8a7d57`、SPARCS `88c925370b5fa08345c1b98578f657b9559eaec8e428348bd6895519ffe49fce`。


---

<a id="phase-64b"></a>

## Phase 64B 实验记录

### 目标

只改变输入光谱配置，检验 RGB 条件下的 MsRE 迁移—保持优势能否在 Sentinel-2 / Landsat 公共六波段下复现。先完成六波段输入协议审计并重训 CloudSEN 源模型；只有 source-val 三类 mIoU 相对 RGB 源模型下降不超过 1 点，才启动 L8 与 SPARCS 的 source-only / MsRE 目标域实验。

### 改动与协议

- 固定六波段顺序：Sentinel-2 `B4/B3/B2/B8/B11/B12`；Landsat-8 `B4/B3/B2/B5/B6/B7`。
- Sentinel-2 反射率按官方缩放读取，连续波段使用 area 重采样到 20 m，标签使用 nearest；Landsat-8 与 SPARCS 保持原 30 m 网格，不宣称两个传感器像元物理尺度一致。
- nodata / 非有限值转零并生成有效性掩膜；饱和值按协议裁剪。归一化统计仅由 CloudSEN train split 计算：mean=`[0.20777850,0.20358557,0.21717624,0.29672984,0.22447699,0.15877002]`，std=`[0.20317571,0.18181795,0.18963913,0.17799714,0.14027748,0.11965738]`。
- 六通道 patch embedding、adapter depthwise 与 pointwise 输入层均显式扩展初始化；未把 RGB checkpoint 伪装为六通道模型。源训练仅更新六波段 stem、原有 CloudAdapter 与同一 readout，DINO blocks 冻结；未增加 head、loss 或 router。
- 输入审计摘要：`src/work_dirs/phase64b_input_protocol/summary.json`，SHA256 `744d9512939bc9c1384a5c8d51c893a3d8397b01180d333b5c12136d78c2afc1`。L8 manifest SHA256 `885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b`；SPARCS manifest SHA256 `7fd78c026d7ee76e3fc8e3dc6f775ddeb86312242650d6ea085502622d05738c`。
- RGB 源 checkpoint SHA256 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`；六通道初始化 checkpoint SHA256 `774b15718b8f211bef361d50c7f24041f283a1f2aaf17a6a30201789de9c97bb`。
- target-test 读取次数为 0；CloudSEN test 数据未下载。L8 与 SPARCS target-train 六通道读取预检通过，但未启动目标域训练。

### 网络与运行情况

- 官方 CloudSEN train / val 六波段约 27 GB 下载完成。首次下载发现 Hugging Face Xet 默认写入用户级缓存后立即终止；随后将全部缓存固定在项目目录内，未删除授权目录外缓存。
- 两次训练前配置错误分别来自配置中暴露 `Path` 类和不可 deepcopy 的文件句柄；均仅在本地修复、验证、push，远端 pull 后重启。失败目录保留用于审计。
- 正式运行使用远端 `conda activate cloud-lite-pt210`，4000 steps，80 个 loss 观测全部有限，范围 `8.4014–20.6970`，无 traceback；完成 8 次验证并生成 `TRAIN_COMPLETE`。

### 止损线

- 64B-1：CloudSEN 六波段 source-val 三类 mIoU 相对 RGB 源模型下降不得超过 1.0 点。
- 只有通过 64B-1 才运行 L8 / SPARCS 的 six-band source-only 与 MsRE；因此目标域增益、Shadow 门和 PEFT 参数门在本轮均未进入判定。

### 真实结果与结论

- RGB 源模型参考 mIoU：`77.8538`。
- stems-only 诊断运行 mIoU：`61.50`，下降 `16.3538` 点；该运行仅用于暴露训练能力不足，不作为正式六波段结论。
- 修正后的正式六波段源模型在 iter 4000 达到最佳 mIoU `58.73`：Surface `74.06`、Cloud `65.49`、Shadow `36.64`；相对 RGB 源模型下降 `19.1238` 点。
- 最佳 checkpoint：`src/work_dirs/phase64b_source_parent/seed64/best_mIoU_iter_4000.pth`，SHA256 `e12714f732e50ec9f2945cc4ec0b8b763fe15992275dab48f2997d65b6727c40`。
- **判定：64B-1 明确失败，按预注册规则停止。** 未运行 L8 / SPARCS 六波段 source-only 或 MsRE，因而没有目标域结果，也不能声称六波段复现了 MsRE 稳健性或带来额外收益。
- 限制：本协议同时把 Sentinel-2 输入落实为原始反射率的统一 20 m 管线，而旧 RGB 参考来自既有 RGB 管线。当前失败说明这套预注册六波段输入配置未保住源性能；若要把损失单独归因于“增加波段”，必须另做同一 20 m / 同一辐射处理下的匹配 RGB 控制，不能用本结果直接归因。


## Phase400：MsRE与输出残差头拆分（2026-10-31）

153 Catalogue fit代表、32development组（7组云影AP），Source CloudSEN val535；RGB/512训练裁块、batch1、FP32、4090D，seed40001，MH/M/H各4000步与每500步val选best，短跑各20+100。Source及BN/buffer冻结，输入／标签／增强／LR逐步三臂配对通过；不读取confirmation/test。

数据结果：Source目标mIoU67.1334；best MH77.6559@2500、M78.1895@1000、H77.7668@4000；final77.4103/72.4039/77.7668。best宏AP69.5126/69.4896/70.0159，池化AP74.1703/63.6909/73.6091；可训练参数380577/354433/26144。主要best差MH−M=−0.5336、MH−H=−0.1109点。Source val535本轮77.8152，开启best MH/M/H分别63.3929/64.5872/66.3939；关闭均精确恢复，不能当作开启路径无遗忘。

实际正式12000、短跑360、门检3更新，总19155前向；含短跑末端96次自动development评价，未用于删臂。GPU阶段1913.389秒，含等待和IO；主进程GPU reserved峰值1.072GiB，CPU RSS峰值2.054GiB（子进程总RSS未全采样）。原CPU统计178.088秒，独立分桶AP／混淆／选择／成本／Source冻结与端点SHA验收通过；一次CPU缺依赖导入失败保留，零额外神经调用，无环境修改。

结论探讨：未观察到MH超过H的mIoU优势；M有训练后期回落，H应作低成本强基线。适配开启的源域下降显著，冻结权重不等同于保住开启路径性能。单seed与已暴露development仅探索，无独立检验；历史Source差异0.0386点未归因。完整结果：outputs/phase400/phase400_20261031/RESULT.md，摘要src/research_results/phase400_20261031/RESULT.md。权重／数组留shared，未同步work_dirs；不自动执行401。

## Phase401：H残差强度诊断（2026-10-31）

固定Phase400 H最佳checkpoint a86d96f3…，四输出残差同时乘α；M关闭，不改权重。只Catalogue development32与CloudSEN val535、原512输入、batch1、FP32、4090D；α0/1先逐图复现Source/H，后0.25/0.5/0.75。

数据结果（α／目标mIoU／源mIoU／目标收益保留%）：0.0/67.1334/77.8152/0.00；0.25/72.3394/75.6669/48.96；0.5/76.8294/72.6227/91.18；0.75/78.0171/69.4310/102.35；1.0/77.7668/66.3939/100.00。实际2835前向、零更新，工作进程179.422秒，GPU reserved峰值0.641GiB；非纯GPU核时间。端点全567样本预测SHA/CM与目标AP精确复现；本地独立逐图混淆、汇总、宏AP算术、身份与账本验收。中间点池化AP未传像元独立重放，验收范围如实保留。

结论探讨：减弱残差的目标收益保留与源损失存在明确取舍，完整五点见outputs/phase401/phase401_20261031/RESULT.md与tradeoff.csv。仅固定H单seed与已暴露池诊断，不用源val选择α后称独立确认；不追加点、不训练、不自动402。权重／数组／work_dirs未传回。


## Phase402 — H输出残差逐层关闭诊断（2026-10-31）

- 目标：固定Phase400 H权重、其余输出残差强度0.5，分别关闭block2/5/8/11输出残差，检验源域损伤与目标收益的层间差异；MsRE关闭，零训练。
- 输入：沿用Catalogue已暴露development32与CloudSEN val535；不读fit/test。all_half基线567张预测与Phase401一致，目标AP复现。
- 结果（目标/源mIoU）：all_half 76.8294/72.6227；drop_2 70.6750/75.2707；drop_5 75.0021/74.3450；drop_8 76.0859/72.2044；drop_11 75.7304/71.6270。
- 结论：无关闭配置同时改善两域；关闭浅层2/5以目标下降换源恢复，关闭深层8/11两域均降。暂不支持删头；若继续，检验少量浅弱深强组合并对照统一缩放曲线，不自动403。
- 执行：使用宿主SSH配置连接gzs，独立新目录，未改旧结果或环境；2835前向、0更新，176.900秒工作进程墙钟（含加载/CPU统计/I/O），峰值reserved687865856字节；原预算1小时，未触发恢复。
- 验收：传输哈希、代码/输入/权重身份、连续调用账本、逐图与池化混淆/IoU/召回/FPR、宏AP算术、基线逐图预测均通过。新点池化AP沿用已验收实现，未传像元，不称独立原像元复算。单checkpoint、条件消融、已暴露评价，不证明稳定性或独立泛化。
- 交付：outputs/phase402/phase402_20261031/RESULT.md、summary.csv、per_group.csv、source_per_image.csv、RUN_INFO.json、INDEPENDENT_ACCEPTANCE.json。


## Phase403 — 浅层减弱、深层保留（2026-10-31）

- 固定Phase400 H权重与Phase401/402评价支持，零训练；先复现all_half，后固定三个分层缩放组合。权重及旧结果不变，不读fit/test。
- 配置按block2/5/8/11排序：all_half=[.5,.5,.5,.5]；weak_2=[.25,.5,.5,.5]；weak_5=[.5,.25,.5,.5]；weak_2_5=[.25,.25,.5,.5]。
- 目标/源mIoU：all_half 76.8294/72.6227，weak_2 74.2004/74.3959，weak_5 76.5545/73.5739，weak_2_5 73.8004/75.3113。
- weak_5相对基线目标−0.2749、源+0.9512点，宏AP+0.2050点，Shadow IoU−0.7822点；只是取舍，不是全面改善。三个组合与Phase401稀疏统一缩放点均互不支配，尚不能证明分层优势。下一步如继续，需实测源损失相近的统一缩放对照，未自动404。
- gzs单卡执行2268次完整前向、0更新，141.499秒工作进程墙钟，峰值reserved687865856字节；原预算1小时，无恢复。主进程峰值RSS3748757504字节。
- 基线567张预测与Phase401一致；传输/代码/输入/权重SHA、连续账本、逐图/汇总混淆和IoU、宏AP算术独立复核通过。新点池化AP未独立原像元复算。单checkpoint、反复使用源val及development，只作探索。
- 完整结果outputs/phase403/phase403_20261031/RESULT.md，含uniform_comparisons.csv及独立验收；不回传权重/像元，不修改历史数据。


## Phase404 — 统一缩放加密对照（2026-10-31）

- 目标：固定H checkpoint补测统一0.40/0.45，检验Phase403 weak_5是否有超出已有稀疏统一网格的线索。先复现统一0.5，零训练，不追加强度。
- 数据：Catalogue已暴露development32与CloudSEN val535；身份/代码/权重与历史一致，旧产物只读，不读fit/test。
- 目标/源mIoU：统一0.5为76.8294/72.6227；统一0.40为75.6202/73.8332；统一0.45为76.2919/73.2402。
- weak_5（76.5545/73.5739）相对统一0.45两域分别+0.2626/+0.3337点；相对0.40目标+0.9343但源−0.2593点。支持保留分层探索线索，不证明优于所有统一强度或跨seed稳健。
- gzs单卡运行1701次完整前向、0更新，110.394秒工作进程墙钟，GPU reserved峰值687865856字节；1小时预算，无失败恢复。未下载或更改环境。
- 基线567张预测复现，调用账本、输入/代码/权重/传输SHA、逐图及汇总混淆与指标、宏AP算术独立验收通过。新点池化AP未独立原像元复算。
- 交付outputs/phase404/phase404_20261031/RESULT.md及weak5_comparisons.csv。建议固定配方在新seed验证，停止本checkpoint继续细搜；尚未启动405。

## Phase405：固定缩放配方跨seed验证（2026-11-01）

- 新seed40501/40502，H独立初始化、采样及增强；冻结Source，仅训练26144参数。复用153 fit、32 development、535源val；每seed短跑120后重新正式4000步，固定最终checkpoint评价六配置。旧40001仅发现参照。
- weak_5目标/源mIoU：40501为75.7501/73.7031；40502为75.9436/74.2441。相对统一0.45分别+0.1637/+0.9809、+0.1679/+0.6737个百分点；平均+0.1658/+0.8273，双域改善方向复现。
- 两seed相对统一0.40亦双域改善；相对统一0.50仍有目标/源取舍。源域仍低于Source 4.1121/3.5711点，未解决损伤。相同已暴露数据上的两个新seed不是独立泛化或显著性证明。
- gzs单卡顺序运行，15620次前向、8240次更新（正式8000、短跑240），控制器1413.8秒；无失败重跑。哈希、调用账本、初始化与数据顺序、冻结回执、逐图/汇总指标、Source与最终H端点通过本地独立验收；新点池化AP未独立像元复算。
- 交付outputs/phase405/phase405_20261101/RESULT.md、summary.csv、paired_deltas.csv及两seed完整小型证据。保留weak_5固定基线，下一步考虑训练期保持约束，不再细搜缩放；源val不进训练。尚未启动406。

## Phase406：训练期残差能量约束（2026-11-01）

- 固定系数1的四层归一化残差能量惩罚；H26144参数，Source冻结。复用405两个seed、初始化、逐步输入/增强/学习率，固定最终4000步。153 fit训练，32 development与535源val仅评价。
- weak_5目标/源差：seed40501 -2.2977/+1.8772；seed40502 -2.8273/+1.5999个百分点（同seed同缩放对405）。
- 1.0目标/源差：seed40501 +0.8675/+0.6378；seed40502 +0.1446/+0.7888个百分点（同seed同缩放对405）。
- 主要weak_5双域改善判据未通过。完整解释见RESULT；两seed及已暴露数据仅作探索。
- 15620次完整前向、8240次更新（8000正式+240短跑），1435.1秒控制器墙钟。参数/冻结/调用/输入身份、逐图汇总指标和端点验收通过；正则梯度自测与实际损失通过；新点池化AP未独立像元复算。
- 交付outputs/phase406/phase406_20261101/RESULT.md、versus_phase405.csv、penalty_summary.csv及原始证据。未启动407。

## Phase407：目标精度对齐后的源域比较（2026-11-01）

- 固定405/406两个seed最终4000步权重，3420前向、零更新；32 development选统一alpha，选定后535源val评价。每模型8次中点探测；目标参照为原405 weak_5，容差0.05点；不追加搜索。
- seed40501：406−405匹配统一缩放目标差+0.0047、源差+0.3350；406−原weak_5源差-0.8307个百分点。
- seed40502：406−405匹配统一缩放目标差-0.0005、源差-0.3310；406−原weak_5源差-1.1587个百分点。
- 主要判据未通过；完整解释见RESULT。目标development参与匹配、源val前轮已暴露，不作独立泛化或显著性证明。
- 控制器511.2秒。输入/权重/代码/传输身份、8步目标选点重放、参考与选点预测复现、调用账本及逐图汇总指标验收通过；新点池化AP未独立像元复算。
- 交付outputs/phase407/phase407_20261101/RESULT.md、matched_summary.csv、paired_deltas.csv及完整小型证据。未启动408。

决策：暂停当前全层能量惩罚及细搜，保留原weak_5固定基线，后续优先考虑分层训练设计；不推广为所有正则无效。

## Phase408：仅block5头的残差约束（2026-11-01）

- 仅block5头的归一化残差能量惩罚，系数0.25，其他三头无正则；H26144参数，Source冻结。复用405两个seed、初始化、逐步输入/增强/学习率，固定最终4000步。153 fit训练，32 development与535源val仅评价。
- weak_5目标/源差：seed40501 -0.3338/+0.3436；seed40502 -0.2101/+0.1955个百分点（同seed同缩放对405）。
- 1.0目标/源差：seed40501 +0.0000/+0.7347；seed40502 +0.1278/+0.6895个百分点（同seed同缩放对405）。
- 主要weak_5双域改善判据未通过。完整解释见RESULT；两seed及已暴露数据仅作探索。
- 15620次完整前向、8240次更新（8000正式+240短跑），1427.0秒控制器墙钟。参数/冻结/调用/输入身份、逐图汇总指标和端点验收通过；正则梯度自测与实际损失通过；新点池化AP未独立像元复算。
- 交付outputs/phase408/phase408_20261101/RESULT.md、versus_phase405.csv、penalty_summary.csv及原始证据。未启动409。

决策：保留405 weak_5，当前单头约束不升级为主方案，暂止能量惩罚系数/层位细搜。下一步建议先核实源训练集，再设计少量源训练重放对照，源val仅评价；尚未执行。

## Phase409：Cloud-Adapter与MsRE并联连接（2026-11-01）

- 回到目标域性能优先。串联M(C(x))对比并联C(x)+(M(x)−x)，block2/5/8/11，H关闭；新增0参数，两臂只训练MsRE354433参数。seed40901，初始化/每步输入/增强/学习率匹配，短跑120后重置正式4000步；153 fit、32 development，每500步评价，best按目标mIoU选，final另报。
- serial best step1000：目标mIoU 77.0094，Shadow IoU 41.2297，宏AP 69.1708。
- serial final step4000：目标mIoU 71.9463，Shadow IoU 45.4499，宏AP 67.4552。
- parallel best step1000：目标mIoU 76.9105，Shadow IoU 39.6851，宏AP 69.5215。
- parallel final step4000：目标mIoU 72.0513，Shadow IoU 45.5681，宏AP 67.8054。
- 并联−串联best目标差-0.0989、final差+0.1050点；预设+0.30点晋级线未达到。历史400仅跨seed参考。本轮单seed、已暴露development，不证明稳定性/独立泛化/创新性。
- 9886次完整前向、8240次更新，控制器1269.5秒。连接公式/梯度/恒等路径、配对身份与轨迹、冻结/参数梯度、调用/选模及逐组汇总指标验收通过。源域仅选模后535图附表；新点池化AP未独立像元复算。
- 交付outputs/phase409/phase409_20261101/RESULT.md、summary.csv、完整TARGET_SUMMARY.csv/TARGET_GROUPS.csv、timing.csv及验收证据。未启动410。

决策：简单并联未达到目标性能晋级线，不替换串联；建议下一轮检验MsRE固定三尺度平均改为输入自适应融合。源域不参与该决策；尚未实现下一轮。

## Phase410：MsRE自适应尺度融合（2026-11-01）

- 回到目标域性能优先。固定三尺度平均对比逐位置softmax尺度融合，block2/5/8/11，H关闭；新增1155参数，原版/改版训练参数354433/355588。seed40901，初始化/每步输入/增强/学习率匹配，短跑120后重置正式4000步；153 fit、32 development，每500步评价，best按目标mIoU选，final另报。
- baseline best step1000：目标mIoU 77.0037，Shadow IoU 41.2633，宏AP 69.1695。
- baseline final step4000：目标mIoU 71.9541，Shadow IoU 45.4666，宏AP 67.4538。
- adaptive best step1000：目标mIoU 76.7918，Shadow IoU 41.5004，宏AP 69.2019。
- adaptive final step4000：目标mIoU 71.9108，Shadow IoU 45.4431，宏AP 67.4998。
- 自适应−固定平均best目标差-0.2119、final差-0.0433点；预设+0.30点晋级线未达到。历史400仅跨seed参考。本轮单seed、已暴露development，不证明稳定性/独立泛化/创新性。
- 9886次完整前向、8240次更新，控制器1383.0秒。等权初始化/梯度/恒等路径、配对身份与轨迹、冻结/参数梯度、调用/选模及逐组汇总指标验收通过。源域仅选模后535图附表；新点池化AP未独立像元复算。
- 训练后汇总存在旧目录名残留，已保留失败记录与原始执行代码，仅恢复CPU报告；新增训练/完整前向均为0。正式初始权重重建及训练后门控非零检查通过。
- 交付outputs/phase410/phase410_20261101/RESULT.md、summary.csv、完整TARGET_SUMMARY.csv/TARGET_GROUPS.csv、timing.csv及验收证据。未启动411。

决策：本轮门控未达到目标性能晋级线，保留原版；源域不参与决策。

## Phase411：MsRE分层残差控制与云影结构建议（2026-11-01）

- 共享残差系数改为block2/5/8/11各一个，初值0.001，净增3参数。原串联、H关闭，seed40901；复用Phase410原版，改版短跑120后重置正式4000步，每500步验证。共有权重、输入、增强和学习率配对验收；源域不作门槛，本轮不重复源域评价。
- baseline best step1000：目标mIoU 77.0037，Shadow IoU 41.2633，宏AP 69.1695。
- baseline final step4000：目标mIoU 71.9541，Shadow IoU 45.4666，宏AP 67.4538。
- layerwise best step1000：目标mIoU 77.4375，Shadow IoU 41.2615，宏AP 68.5388。
- layerwise final step4000：目标mIoU 71.2480，Shadow IoU 44.6936，宏AP 66.6828。
- 分层−原版best差+0.4338，final差-0.7061点；预设+0.30点筛选线达到。单seed且development参与选模，不能称稳定或独立泛化。
- best总体增益来自Surface/Cloud；Shadow IoU 41.2633→41.2615，宏AP 69.1695→68.5388，不能称云影改善，后期回落仍在。
- 新增4408次前向、4120次更新，墙钟610.0秒；历史对照不重复计费。初始输出等价、分层独立梯度、参数/输入身份、轨迹、调用及指标验收通过。
- 另交CLOUD_SHADOW_ARCHITECTURE.md：优先检验云遮挡与地表遮阴双解释，备选为未匹配可回退的云影位移关联、跨地物照明场。附目标错误构成、代码接入点、已有方法重叠及最小消融；均为待测方案，不宣称首创。本轮未启动这些结构训练。
- 交付outputs/phase411/phase411_20261101/RESULT.md及完整曲线、参数系数、调用与验收。

## Phase412：三种动态MsRE机制（2026-11-01）

- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。
- region_fixed：mIoU 73.1420，影IoU 42.6151，影宏AP 68.5701。
- region_dynamic：mIoU 72.8913，影IoU 42.9096，影宏AP 68.7791。
- layer_fixed：mIoU 71.8573，影IoU 43.2767，影宏AP 67.9357。
- layer_dynamic：mIoU 71.2952，影IoU 43.7577，影宏AP 67.5847。
- site_fixed：mIoU 76.5601，影IoU 42.6149，影宏AP 68.9725。
- site_dynamic：mIoU 71.5351，影IoU 43.4818，影宏AP 67.7389。
- 晋级4000步：无，未同时超过对应所有对照及Phase411至少0.30点。单seed短程筛选，不代表最终上限或稳定性。
- 有效运行6384次前向、6000次更新；另有保留的修正尝试2061次已记录前向、2000条已完成更新记录。详见验收中的原因和远端归档。
- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。
- 交付outputs/phase412/phase412_20261101/RESULT.md及指标、代码锁、调用和验收记录。

## Phase413：云影双算子、位移关联与共享照明场（2026-11-01）

- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。
- dual_generic：mIoU 73.3577，影IoU 42.6656，影宏AP 68.8258。
- dual_operator：mIoU 74.0174，影IoU 42.2379，影宏AP 68.7956。
- dual_shuffled：mIoU 74.4534，影IoU 42.1147，影宏AP 68.8606。
- link_generic：mIoU 72.4113，影IoU 43.3678，影宏AP 68.8416。
- link_cloud：mIoU 73.1729，影IoU 42.9585，影宏AP 68.8492。
- field_spatial：mIoU 73.4256，影IoU 42.3448，影宏AP 68.6516。
- field_illumination：mIoU 73.4827，影IoU 42.2026，影宏AP 68.6211。
- field_shuffled：mIoU 73.3194，影IoU 42.3750，影宏AP 68.7000。
- 晋级4000步：无，未同时超过对应所有对照及Phase411至少0.30点。单seed短程筛选，不代表最终上限或稳定性。
- 有效运行8512次前向、8000次更新；另有保留的修正尝试0次已记录前向、0条已完成更新记录。详见验收中的原因和远端归档。
- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。
- 所有对照同样获得原始RGB和语义辅助损失；只将组内差值用于机制判断。物理算子、软参考及位移均是代理机制，未反演真实物理参数。
- 交付outputs/phase413/phase413_20261101/RESULT.md及指标、代码锁、调用和验收记录。

## Phase414：辅助语义监督单独诊断（2026-11-01）

- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。
- aux_only：mIoU 74.6427，影IoU 41.8732，影宏AP 68.7957。
- 晋级4000步：不适用，本轮仅诊断。单seed短程筛选，不代表最终上限或稳定性。
- 有效运行1064次前向、1000次更新；另有保留的修正尝试0次已记录前向、0条已完成更新记录。详见验收中的原因和远端归档。
- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。
- 仅增加同款语义头与0.1 CE，无新结构残差；原始RGB沿用流程但不参与辅助头。与Phase411及Phase413的差值见COMPARISONS.csv。
- 交付outputs/phase414/phase414_20261101/RESULT.md及指标、代码锁、调用和验收记录。

## Phase415：辅助语义监督 detach 诊断（2026-11-01）

- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。
- aux_detached：mIoU 73.1470，影IoU 43.3737，影宏AP 69.1917。
- 晋级4000步：不适用，本轮仅诊断。单seed短程筛选，不代表最终上限或稳定性。
- 有效运行1064次前向、1000次更新；另有保留的修正尝试0次已记录前向、0条已完成更新记录。详见验收中的原因和远端归档。
- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。
- 同款语义头与0.1 CE，输入 detach 阻断辅助梯度，无新结构残差；原始RGB沿用流程但不参与辅助头。与Phase411及Phase414的差值见COMPARISONS.csv。
- 交付outputs/phase415/phase415_20261101/RESULT.md及指标、代码锁、调用和验收记录。

## Phase416：detach 加独立梯度裁剪诊断（2026-11-01）

- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。
- aux_separate_clip：mIoU 77.3958，影IoU 41.1275，影宏AP 68.4701。
- 晋级4000步：不适用，本轮仅诊断。单seed短程筛选，不代表最终上限或稳定性。
- 有效运行1064次前向、1000次更新；另有保留的修正尝试0次已记录前向、0条已完成更新记录。详见验收中的原因和远端归档。
- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。
- 同款语义头与0.1 CE，输入 detach，语义头与MsRE分别裁剪梯度，无新结构残差；原始RGB沿用流程但不参与辅助头。与Phase411及Phase415的差值见COMPARISONS.csv。
- 交付outputs/phase416/phase416_20261101/RESULT.md及指标、代码锁、调用和验收记录。

## Phase417：修正训练后的云影双算子对照（2026-11-01）

- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。
- dual_generic：mIoU 77.0538，影IoU 41.0135，影宏AP 68.7110。
- dual_operator：mIoU 76.8144，影IoU 41.4690，影宏AP 68.9359。
- dual_shuffled：mIoU 75.6508，影IoU 43.8260，影宏AP 69.1933。
- 晋级4000步：不适用，本轮仅诊断。单seed短程筛选，不代表最终上限或稳定性。
- 有效运行3192次前向、3000次更新；另有保留的修正尝试0次已记录前向、0条已完成更新记录。详见验收中的原因和远端归档。
- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。
- 同款语义头与0.1 CE，输入 detach，语义头与MsRE分别裁剪梯度，复测通用、双算子与打乱参考三组；与Phase411/416及组内对照的差值见COMPARISONS.csv。
- 交付outputs/phase417/phase417_20261101/RESULT.md及指标、代码锁、调用和验收记录。

## Phase418：修正训练后的云影关联与照明场对照（2026-11-01）

- 153 fit / 32 development，seed40901；以Phase411分层MsRE为底座，H关闭。每组1000步，每500步验证，仅第1000步筛选；源域不参与选择。
- link_generic：mIoU 74.1953，影IoU 44.1572，影宏AP 69.2482。
- link_cloud：mIoU 74.1859，影IoU 44.1592，影宏AP 69.2469。
- field_spatial：mIoU 76.5302，影IoU 41.7385，影宏AP 69.0173。
- field_illumination：mIoU 76.2588，影IoU 43.1693，影宏AP 69.1660。
- field_shuffled：mIoU 76.7951，影IoU 42.8423，影宏AP 69.1390。
- 晋级4000步：不适用，本轮仅诊断。单seed短程筛选，不代表最终上限或稳定性。
- 有效运行5320次前向、5000次更新；另有保留的修正尝试0次已记录前向、0条已完成更新记录。详见验收中的原因和远端归档。
- 数据/增强/学习率配对、初始权重、冻结权重、调用、逐组混淆矩阵与精确AP、checkpoint身份及晋级规则验收通过。
- 同款语义头与0.1 CE，输入 detach，语义头与MsRE分别裁剪梯度，复测云影关联两组和照明场三组；与Phase411/416及组内对照的差值见COMPARISONS.csv。
- 交付outputs/phase418/phase418_20261101/RESULT.md及指标、代码锁、调用和验收记录。

## Phase419：零训练错误诊断（2026-11-01）

- 11模型、固定32组、352份已有得分；零训练、零新增前向，CPU诊断及本地独立数值验收通过。
- 阴影IoU是基线瓶颈；三个结构的影类均改善，但云/地表损失抵消收益。共同掉点高度集中于T25XEG高亮低纹理、全地表真值场景。
- T40XDR雪冰场景基线漏掉全部85502个阴影；新结构部分找回，但伴随地表误报云。
- 场景替换、真值分层仅诊断；不删场景、不改标签，不把反事实作为成绩。建议先核查标签来源，再试保持非阴影云/地表比值的阴影专用分支；未执行。
- 结果：outputs/phase419/phase419_20261101/RESULT.md。

## Phase420：冻结目标基线的阴影概率修正（2026-11-01）

- 固定Phase411第1000步基线，训练6817参数阴影分支；整图/空间两组各追加1000步。153 fit/32 development标签不变；seed40901。
- global: mIoU 78.2964、影IoU 43.8816、影宏AP 68.5388。
- spatial: mIoU 78.5333、影IoU 44.6030、影宏AP 68.5830。
- 空间方案余量+0.2369点，通过0.30门槛：False。所有评价地表↔云直接切换为0，基线预测复现及冻结验收通过。
- 2192次前向、2000次更新；预检配置类型失败零前向并归档。单seed结果，不自动延长训练。
- outputs/phase420/phase420_20261101/RESULT.md。

## Phase421：MsRE局部材质参考三组对照（2026-11-01）

- 三组各1000步、seed40901、153 fit/32 development；原始Source起步，不用Phase420修正，不加辅助损失。
- baseline：mIoU 77.4051、影IoU 41.1499、影宏AP 68.4615。
- neighborhood：mIoU 72.4558、影IoU 43.4288、影宏AP 68.8769。
- material：mIoU 74.4781、影IoU 44.4429、影宏AP 69.2532。
- 材质版相对最强对照 -2.9270个百分点；超过两对照：False。单seed筛选，不自动追加训练。
- 3192次前向、3000次更新，配对/冻结/完整预测重算/原标签一致验收通过。
- outputs/phase421/phase421_20261101/RESULT.md。

## Phase422：材质参考分支开关损伤诊断（2026-11-01）

- 同一Phase421材质检查点，开启74.4781%、关闭77.2775%，独立基线77.4051%。关闭仅绕过新增路径，保留已共同训练的MsRE。
- 主要整体损失来自当前分支直接作用；共同训练仍留有高亮场景局部损伤。不能把端点差值当成完整训练因果分解。
- 正式64次前向零训练，原预测/分数精确复现、权重与标签不变、独立混淆矩阵复核通过；失败尝试另计2次前向。
- outputs/phase422/phase422_20261101/RESULT.md 与 INTERPRETATION.md。

## Phase423：材质残差零训练强度扫描（2026-11-01）

- 同一Phase421材质检查点，新增残差alpha=0/0.25/0.5/0.75/1；原MsRE与标签不变。
- alpha=0：mIoU 77.2775%，影IoU 43.5549%，影宏AP 69.2865%。
- alpha=0.25：mIoU 76.4752%，影IoU 43.7843%，影宏AP 69.2786%。
- alpha=0.5：mIoU 75.7031%，影IoU 44.0553%，影宏AP 69.2548%。
- alpha=0.75：mIoU 74.9567%，影IoU 44.2676%，影宏AP 69.2618%。
- alpha=1：mIoU 74.4781%，影IoU 44.4429%，影宏AP 69.2532%。
- 160次前向、零训练，两端点预测和分数逐位复现；开发集网格最佳相对独立基线 -0.1276点，不能当作独立确认。
- outputs/phase423/phase423_20261101/RESULT.md。

## Phase424：材质参考与残差机制诊断（2026-11-01）

- 原模型 74.4781%、自参考 74.4831%；参考=query且差异项清零、保留原门控，输出几乎不变。
- T25XEG参考为地表类，明显更亮参考权重仅0.0730%，门控仍0.3757；当前显式关系通道作用很小，绝对外观修正缺乏零差异零响应约束。
- 64次前向零训练，32组预测/分数逐位复现，逐格统计和本地独立验收通过；不声称同材质真值或完整因果证明。
- outputs/phase424/phase424_20261101/RESULT.md 与 INTERPRETATION.md。

## Phase425：材质残差归零与亮度支持约束（2026-11-01）

- 三组各1000步、seed40901、原损失与完整32组；不加末端修正，不改标签。
- material_control：mIoU 74.5287%，相对Phase421独立MsRE基线 -2.8763点。
- centered：mIoU 76.4463%，相对Phase421独立MsRE基线 -0.9588点。
- centered_supported：mIoU 77.2713%，相对Phase421独立MsRE基线 -0.1338点。
- 3192次前向、3000次更新；归零/门控/初始化测试、逐步输入与LR配对、冻结及本地指标复核通过。归零组384个输出bias代数抵消。单seed筛选，不自动延长。
- outputs/phase425/phase425_20261101/RESULT.md。

## Phase426：三类语义token与竞争路由（2026-11-01）

- 完整方案直接训练，无前置特征探针；153 fit/完整32 development，seed40901，各1000步。
- baseline：mIoU 77.2648%，三类IoU [95.2335195839416, 95.65594861661081, 40.90479959358368]。
- competitive：mIoU 77.0332%，三类IoU [94.48556634349377, 94.96968749626133, 41.64442800587488]。
- 同轮净变化 -0.2315个百分点；新增15参数、0.05语义辅助损失及在线困难度权重，不能单独归因架构。
- 2128次前向、2000次更新；共同初始化/逐步输入与LR/冻结/预测与AP复算通过。单seed开发集探索。
- outputs/phase426/phase426_20261101/RESULT.md。

## Phase427：Cloud-Adapter内融合MsRE与局部细节反馈（2026-11-01）

- 三组各1000步；153 fit/完整32 development、seed40901，原分割损失；新增较高分辨率局部读取和语义反馈，无辅助监督。
- baseline：mIoU 77.4356%，三类IoU [95.29143053010809, 95.76258450597439, 41.25284730524762]。
- unfrozen：mIoU 57.3010%，三类IoU [63.298836411471385, 74.16962767609682, 34.43447699885003]。
- integrated：mIoU 55.5799%，三类IoU [63.1106362812603, 73.54302254177547, 30.086096208780816]。
- 融合相对基线 -21.8557点，相对仅解冻 -1.7211点。
- 3192次前向、3000次更新；共同初始化、输入/LR配对、指定冻结范围、完整预测与AP复算通过。后两组允许Source交互层更新，不能称全Source冻结。单seed开发集探索。
- outputs/phase427/phase427_20261101/RESULT.md。

## Phase400—427训练协议审计附录（2026-11-01）

- 仅审计，零训练/前向；不改旧Phase封存结果及标签。覆盖28轮，本地49份正式轨迹82000行、3877项清单哈希复核。
- 413—415辅助优化混杂已有416—418纠正；412起大多1000步筛选，不能代表充分收敛上限；426结构与辅助监督未拆分；427只验证局部适配，不能否定完整融合网络重训。
- 早期源域保持判据与当前目标优先不同，406全强度H目标收益作为候选保留。完整逐轮保留/限定/复验建议见 outputs/audit_phase400_427/20261101/RESULT.md。
- 本次未重载远端权重/像元数组，历史数值验收与本次协议审计范围分开说明。

## Phase428 融合网络40k重训启动（2026-11-01）

- 用户授权完整重训；从DINOv2预训练初始化，联合训练CloudAdapter CNN/交互、MsRE、局部细节反馈和三类Mask2Former头，DINO冻结。
- CloudSEN训练8490/验证535，batch2，AdamW 1e-4，warmup500、PolyLR至40000；每2000步验证。
- 远端PID1967549；100步已确认五组模块更新、DINO不变、优化器范围精确。任务运行中，无最终指标。
- 计划：research_plans/phase428.md；运行记录：../outputs/phase428/phase428_20261101/RESULT.md。

## Phase428 40k重训完成查询（2026-11-02）

- 完成40000步，退出码0；20次CloudSEN官方源验证，best=final=78.28% mIoU，地表/云/影IoU=88.25/83.87/62.71%。
- 五组模块实际更新，DINO冻结未变，最终LR归零；运行约3小时3分。完整记录见 ../outputs/phase428/phase428_20261101/RESULT.md。
- 对历史源验证77.8538%数值高约0.43pp，初始化/训练协议不同，不能归因结构净收益；Catalogue尚未评价。

## Phase429 DINOv3-S+/16最小两版实现（2026-11-02）

- 独立实现baseline/fused，mmengine训练、共享Mask2Former头；相同初始化与40k协议。计划见 research_plans/phase429.md。
- 官方S+随机初始化架构GPU结构验收通过：512/batch2损失、反传、预测、冻结梯度及共同初始化。真实预训练权重仍待授权下载，未启动正式训练。
- 结果与代码哈希：../outputs/phase429/phase429_20261102/RESULT.md、CHECKS.json。

## Phase429 正式训练启动（2026-11-02）

用户提供DINOv3-S+/16 safetensors，传输SHA256一致；严格转换并完成两版10步真实数据检查。主干保持预训练值且冻结，共同初始化和输入一致，任务分支均更新。13:50北京时间启动baseline→fused顺序队列，各40k，正式训练不继承检查权重。当前无正式性能结论。记录：`outputs/phase429/phase429_20261102/RESULT.md`、`START_CHECKS.json`、`LAUNCH.json`。

## Phase429 Catalogue评测完成（2026-11-02）

两版CloudSEN最佳权重直接推理固定32 development组，零目标训练。对照mIoU 67.9437%，融合70.1266%（+2.1829点）；云影IoU 21.6415%→24.5073%，宏AP 71.6213%→71.0876%，池化AP 46.3222%→53.1221%。64次正式前向，输入哈希、严格加载、混淆矩阵/AP复算通过。完整记录：`outputs/phase429/phase429_20261102/RESULT.md`及`catalogue_best/`。

## Phase430 原版DINOv2 CloudAdapter＋MsRE启动（2026-11-02）

原CloudAdapter＋SparseMsREResidual在CloudSEN三分类源训练40k，DINO冻结，其余联合训练；seed42901/batch2。前10步四组任务模块均更新，冻结主干不变。完成后自动用源best评测Catalogue32。记录：`outputs/phase430/phase430_20261102/RESULT.md`。

## Phase432 concat投影实验（2026-11-02）

按用户要求，基于Phase431仅替换五尺度池化求和为320→64的无bias 1×1投影，新增20,480参数，五块单位矩阵初始化使初始功能等价求和。同协议全新40k，与现有实验并行，源best自动评价Catalogue32。运行验收和结果见 `outputs/phase432/phase432_20261102/`。

## Phase433 四模型目标域全量微调启动（2026-11-03）

429/430/431/432各自源best初始化，在Catalogue153 fit上全网络解冻训练4000步，32 development每500步评价。主干1e-5、其他1e-4，batch1/seed43301，四组并行。首10步DINO和任务模块实际更新，初始权重严格匹配。结果表自动输出到远端phase433_20261103；本地记录`outputs/phase433/phase433_20261103/RESULT.md`。

## Phase434 431下降与无标签校准诊断完成（2026-11-03）

3500→4000末500步mIoU81.4805→74.8517，云影IoU51.7091→47.1207。4000权重＋3500BN统计恢复81.5439/51.9133；4000权重＋153fit无标签RGB逐层BN重估达到83.6073/57.4313（仍是目标监督模型）。真正无目标监督源模型校准69.2132→69.7756，仅+0.5624点。7组224分割前向、3060空间CNN统计前向、零梯度更新；三端点混淆矩阵精确复现。结果：`outputs/phase434/phase434_20261103/RESULT.md`。

## Phase435 输出残差与VFM奇异更新诊断（2026-11-03）

431最终＋校准BN83.6073%；关闭输出残差83.5899%，回退源残差83.5677%；回退源VFM82.6659%，再关闭残差82.6492%。这是保留目标监督任务模块的推理干预，不是无监督成绩或从头消融。60矩阵SVD更新范数比0.3931%，右奇异最大/最小四分位能量29.75%/17.03%，不支持小奇异分量主导。224前向零训练，端点复现与能量守恒通过。记录：`outputs/phase435/phase435_20261103/RESULT.md`。

## Phase436 目标域监督比例（2026-11-03，已完成）

Phase431源权重初始化，153 fit组固定哈希嵌套选15/38/77组，分别约10%/25%/50%监督。沿用Phase433全参数4000步、seed43301、batch1、LR主干1e-5/其他1e-4。三组并行启动，源权重逐张量匹配、优化器覆盖及第10步DINO/task实际更新均通过。结束后统一153 fit RGB校准BN并评价固定32 development；主表对比0%与100%匹配端点，报告mIoU/云影IoU及跨域增益。远端 /home/scv/shared/phase436_20261103；本地 outputs/phase436/phase436_20261103。单seed，三组4000步均完成，初始化、参数更新及无标签BN校准检查通过。最终校准mIoU：10%=76.0745，25%=81.7339，50%=81.4335；相对0%同BN基线69.7756分别+6.2989/+11.9583/+11.6579点。100%参考83.6073，25%保留86.46%全监督收益。50%略低于25%（0.3004点），本轮不能认定标签增加导致退化。完整结果见outputs/phase436/phase436_20261103/RESULT.md。

## Phase437 25%监督抽样复现（2026-11-03，运行中）

复用Phase436抽样43601的38组结果81.7339%，新增抽样43602、43603，各取38/153 fit组。固定训练seed43301及Phase436其余协议（Phase431源权重、全参数4000步、153 fit RGB BN校准、32 development评价），两组并行。与原子集分别重叠8/12组，两新子集重叠4组。源权重精确加载、优化器覆盖及第10步主干/task实际更新通过。完成后汇总三抽样的mIoU、云影IoU、相对同BN零监督增益、全监督收益保留及均值±样本标准差。本轮测抽样敏感性，训练seed固定。远端/home/scv/shared/phase437_20261103；本地outputs/phase437/phase437_20261103。

## Phase438 10%监督地表损伤五臂诊断（2026-11-03，运行中）

用户授权五臂并行，各4000步、同Phase436原15组、训练seed43301。A原协议，B学习率1/10，C冻结VFM，D冻结且学习率1/10，E附加0.1源监督回放。第10步验证：A/B/E主干198参数张量与task197张量更新；C/D主干0张量更新、task197更新；E源梯度非零且源前向不改变BN运行统计。自动每500步保存模型权重并评价校准后的目标训练/开发及源验证表现，源BN另用153 source train RGB校准。宿主I盘约2.2GB可用，产物通过shared软链接存到远端D盘/mnt/d/CloudAdapterExperiments/phase438_20261103，避免占用I盘；冻结臂不重复保存encoder，无优化器快照。五臂总显存启动约11.7GB；Phase437此时训练已结束。五臂训练完成，目标最终校准mIoU A/B/C/D/E=76.4821/82.7248/79.9670/82.9974/78.1493。D地表95.9433、云96.8794、影56.1695；低LR在全参数和冻结两侧均改善，源回放未恢复地表。诊断曾因序列化dataset.data_list为空中断，已改get_data_info并恢复评测，不重训。完整曲线及源验证仍在运行；目标结果见outputs/phase438/phase438_20261103/FINAL4000_RESULT.md。

## Phase439 同图同标签外观干预（2026-11-03，已完成）

用户指定两权重（Phase431源、Phase433/431全目标监督最终权重）固定既有Phase434 BN，Catalogue固定32组配对测试曝光/对比度/红蓝相对增益/锐化，共16条件、1026次前向，零训练及零BN更新。输入确认线性TOA RGB×255，实测有效值4.6665–587.4945；不做gamma或255上截断，下截断及新增超过反射率1分开计数。记录3×3×3错误转移、空间CNN输出和四次交互前后特征L2/余弦；特征hook及输入管线完全一致性已通过。输出存远端D盘/mnt/d/CloudAdapterExperiments/phase439_20261103。计划见src/research_plans/phase439.md。正式复验1088次前向（加首轮1026，共2114），参数/BN哈希未变，64次hook开关检查逐像素完全一致；历史端点mIoU误差0.000004/0.000092点。温和档适配模型最大下降0.67点，源模型8.22点，适配优势仍为13.31–21.39点。强偏蓝0.5时适配mIoU下降16.17点、原正确地表→云占全部地表25.79%，第一阶段完整交互扰动放大比中位数1.436（源模型1.591）；无实际数值截断。未出现温和外观变化导致适配优势消失的模式。完整主表、曲线、错误转移与特征统计已交付outputs/phase439/phase439_20261103，解读见READOUT.md。

## Phase440 偏蓝加密曲线与第一阶段干预（2026-11-03，已完成）

两权重×12档（原图及0.70到0.50步长0.02）×3路径×32组，2304次前向，零训练、零BN更新，约534秒。常规、关闭第一次交叉注意力、同图原图context替换；其余MsRE/局部读写保留。64张原图替换预测及得分差严格一致；历史三个端点误差<0.001个百分点；权重/BN哈希不变，72组池化指标独立重算通过。适配模型强偏蓝mIoU常规67.4401，关闭76.7692，原图context76.7201；原图分别83.6072/83.6487/83.6072。落差改善9.2876/9.2799点，未损失原图能力。三处集中下跌0.68→0.66、0.58→0.56、0.52→0.50，分别2.96/3.18/3.84点；主要翻转峰由不同场景主导。3977685个最终地表变云像素的平均地表−云得分差原图+0.963至0.5−0.829，稳定地表仍+0.961。证据支持第一条空间注意力注入参与强偏蓝损伤；恢复主要为地表/云，云影IoU几乎未变。关闭分支原图+0.0415点不足以认定普遍收益；原图context是诊断参照。输出及曲线、逐图轨迹见outputs/phase440/phase440_20261103/READOUT.md；远端D盘保存。

## Phase441 类条件域偏移与分组线性探针（2026-11-03，已完成）

按用户协议执行三模型：Phase431 Source、100%目标适配（固定Phase434 BN）、官方DINOv3-S+/16；535源图按107 ROI，目标32组。1701次冻结骨干前向，约126秒；网络零更新。32×32 token原始掩膜精确面积对应，纯度/有效比例≥90%，每场景每类最多128 token，混合patch另列。提取四阶段pre/attention/post及最终/CNN特征，阶段内共享域/场景/位置均衡64维PCA（解释方差73.69%–100%），原维R复核；256次场景bootstrap、64方向SW1及精确复核、100次同域匹配分组、525个训练折内PCA/标准化的逻辑回归探针。统计约99秒。567图Source首交互前与官方原特征严格一致，1701特征签名和525折无场景交叉验收通过。

结论：云影R最大仅部分支持，Source最终云0.769高于影0.726；官方与适配最终影分别1.070/1.077。Source第一次完整交互云域可读准确率85.69→88.96（+3.28点，区间1.51–6.05），云影R反而0.853→0.829；方向并非统一朝混淆类。适配最终地表/云/影R=0.834/0.701/1.077，对比Source0.622/0.769/0.726，域准确率97.18/97.52/74.75，高于Source88.40/96.01/63.52；不支持普遍域不变性改善。最终地表→云P从0.020到−0.116，云影→云0.004到−0.178，提示区分位移大小与方向。目标纯云影仅5组161token，覆盖原始云影24.42%；同域自然波动大。混合patch另列105行。官方最终云影探针49.82%实际为源99.65%、目标0%，不能当作域不变证据。保留原RGB管线及无标签目标fit BN，未将这些因素单独分离。

主表105行及两图、区间/抽样/签名见outputs/phase441/phase441_20261103/READOUT.md。原特征和每折探针参数在远端D盘；本地保存共同PCA、bootstrap、折分、OOF及签名。Phase440干预证据仍成立，本轮不把所有域信息都视为损伤。未启动后续训练。

## Phase442 固定官方DINOv3最终层与唯一PCA（2026-11-03，已完成）

用户要求只比较全局及三类条件距离。固定官方预训练DINOv3-S+/16，第12块后最终LayerNorm特征；源535图/107 ROI、目标32组，保留既有RGB输入。全局用全部580608有效token按域/场景等权拟合唯一64维PCA，解释方差88.3366%，无白化；三类沿用Phase441纯度≥90%固定抽样，全部四行复用同一PCA和64切片方向。新增567次冻结前向75.30秒，统计31.55秒，无探针、无网络更新。旧抽样token567图精确复现，四项F独立SVD复算和全部特征SHA通过。

全局/地表/云/云影F=8.267591/9.490893/8.532731/18.693957，精确切片W1=0.219879/0.229651/0.200651/0.317031。云影条件距离两项均最高；目标纯云影仅5组161token，覆盖24.42%原始影面积，不能外推所有影。256次场景bootstrap固定PCA；云F百分位下限8.6054略高于原点值，原样保留并说明非线性bootstrap偏移，不作显著性结论。全局含混合边界且不类均衡，不是三类距离加权和。主表、唯一PCA、各范围矩及区间在outputs/phase442/phase442_20261103/READOUT.md；全量token在远端D盘。启动前两处脚本语法问题修复，未产生额外网络前向。

## Phase443 整体颜色亮度对齐，固定DINOv3与原PCA（2026-11-03，已完成）

仅增加一次输入外观对照：每域RGB通道4097档分位数单调映射到两域分位数均值，每域所有图共用映射，无标签、无逐图调整；拟合用512输入固定4步长网格，每图16384像素，源ROI均5图/目标每组1图保证场景等权。保留浮点范围，不额外clip；拟合范围外端点映射比例最大约0.000894%。原官方DINOv3-S+/16、block12最终LayerNorm、Phase442原64维PCA字节不变，固定抽样/权重/256场景bootstrap/64切片方向。567次前向，零网络更新。权重状态与原PCA/索引签名、全部特征SHA、四项F独立SVD复算通过。

全局/地表/云/云影F：8.267591→7.517935（−9.07%）、9.490893→8.347497（−12.05%）、8.532731→8.186274（−4.06%）、18.693957→18.761341（+0.36%）。切片W1：0.219879→0.208433、0.229651→0.213333、0.200651→0.186308、0.317031→0.315407。配对F差95%场景区间分别[−1.491,−0.275]、[−2.644,−0.412]、[−1.017,0.007]、[−2.129,1.629]；颜色映射及PCA固定，不包括映射估计不确定性。

输入RGB近似分布W1从19.896/18.938/23.770降至0.457/0.328/0.280；加权RGB亮度从19.121降至1.000。RGB均值差从16.963/12.090/14.829降至−0.365/−0.340/−0.355（0–255输入单位，未截断大于255的原浮点值）。颜色分布已大幅接近，但云影特征距离基本不变。该对照不能把全局类别构成影响与纯成像因素分开，也不保证RGB联合分布一致；不能据此推出分割IoU变化。目标纯云影仍仅5场景161token。完整统计/映射/PCA与复算文件见outputs/phase443/phase443_20261103/RESULT.md，全量特征留远端D盘。启动时本地编码及命令长度问题已修复，未产生额外前向或其他实验。

## Phase444 双域颜色对齐后完整源域重训（重建版，已完成）

用户要求删除双端旧Phase444并按真实意图重做。旧本地outputs/phase444、旧脚本与计划及日志段，远端/mnt/d/CloudAdapterExperiments/phase444_20261103及共享链接均已删除；重新建立本阶段。新远端根目录/home/scv/shared/phase444_retrain_20261103链接到D盘，同名本地目录位于outputs/phase444。

429 baseline与431 fused各从官方DINOv3预训练权重、新任务分支初始化，按原batch2、seed42901、AdamW1e-4、40k协议并行训练。8490张CloudSEN训练图，原随机裁剪/翻转/光度增强后施加Phase443固定源颜色映射，浮点结果进入原归一化；验证/目标推理施加对应域同一固定映射。只用对齐CloudSEN535图源验证选best，Catalogue32组标签仅最后评价。目标标签监督比例0%；不做目标BN拟合，均用各自源训练原生BN。映射已使用两域评测图无标签颜色统计，属于利用目标外观统计的零标签实验。DINO保持原冻结策略，全部任务模块联合重训，不接续旧分割检查点。

启动验收：567张对齐输入与Phase443张量SHA全量一致；训练增强后映射8图检查通过、标签不变。两模型共同初始化和前10批数据一致；10/100步各任务模块参数更新、预训练DINO状态未变，优化器参数集合精确。两轮已超过100步。完成40k后控制器自动比较原源训练权重/原图与新对齐重训权重/对齐图，原对照也使用原生源BN，报告三类IoU/mIoU及权重/输入签名。未产生最终结果，不能引用旧已删除Phase444分数作为本阶段结论。

完成核查：两轮均40000步，耗时429=8070.87秒、431=8318.74秒，并行运行；40k各任务分支更新与DINO冻结验收通过，控制器退出0。全部逐图混淆矩阵本地独立汇总与IoU复算通过，正式文件传输SHA通过。429源mIoU78.733398→78.061117，目标67.943189→64.804248（−3.138941点）；431源79.352778→78.650069，目标69.213169→68.401734（−0.811435点）。431目标地表95.383362→92.950393（−2.432969点）、云89.932303→88.333167（−1.599136点）、影22.323841→23.921641（+1.597799点）。429目标影21.640021→19.542464。重训后本轮仍未观察到整体跨域mIoU收益；431云影收益被其他两类损失抵消。0%目标标签监督，未进行目标BN校准，基线与新模型均为源训练原生BN。结果位于outputs/phase444/phase444_retrain_20261103/RESULT.md；属于单seed固定development比较。

## Phase445 云影70%目标的错误来源诊断（第一步完成）

冻结原431 best32k、原始CloudSEN535图/107 ROI、原生源BN、0%目标标签监督。复用重建Phase444原训练预测，零新增前向/训练，混淆矩阵与独立复测完全一致。云影TP9418493、FN2926601、FP2328375；只修漏检需852936像素、只减误报需1218479像素到70%。FN中地表1267470、云1659131；FP中地表1303245、云1025130。52.73%漏检在真实影边界内3px、75.06%在8px；≤64px碎片仅贡献2.37%，≤256px贡献10.27%。边界外3px误报1050848像素，占误报约45.13%。几何窄影不能等同低遮光薄影；水体/暗云未作无证据自动归类。按每错误方向最大两个不同ROI生成8组图像/真值/预测/错误对照。

建议基于真实1/4空间细节、原RGB对比、多级语义实现候选与环形邻域交互，云上下文为软提示；冻结431和BN，仅训练零初始化云影有符号残差与非零门控。三类监督加正确非影保持约束，只用源训练标签；硬验收影≥70%、地表≥89.1274%、云≥84.7437%。当前仅完成诊断和执行计划，未启动新增分支训练。结果与样例见outputs/phase445/phase445_20261106/RESULT.md。

### Phase445 新分支训练及独立复测完成

用户授权继续到一套新网络出完整结果。实现LocalOcclusionBranch：真实1/4空间CNN细节投影24维、四级384维语义投影16维；7/3、15/7、31/15环形参考，原地表概率+0.1软加权、内容余弦相似度聚合；RGB对数对比、色度、纹理与全局语义；原图分辨率零初始化有符号影logit残差，sigmoid门控初始非零，残差限幅4。仅新增37002参数。原431 best32k及全部BN保持eval冻结，其他两类分数不变。源8490图，随机水平/垂直翻转、不调色，batch4、seed44501、6000步、AdamW峰值2e-4、200步预热＋衰减，三类难例加权CE＋2×正确非影KL＋2×非影softIoU保持＋0.01残差平方。验证535图每1000步完整评估；无目标图像/标签训练。训练含7轮验证391.39秒、27741图前向，再独立单张535图复测。

最佳云影与最佳保护均6000步。独立地表IoU89.127405→89.155129（+0.027725），云84.743702→84.884906（+0.141204），影64.187228→64.889096（+0.701868），mIoU79.352778→79.643044（+0.290266）。保护通过，70%主目标未达，尚差5.110904点。影召回76.293409→77.449017。找回原漏影264678，同时新增漏影122017，净142661；消除原误报222114，同时新增283253，误报净增加61139。净漏影改善134846来自原判云，7815来自原判地表。正确非影像素仍有损伤：地表100710、云138588；保持整体IoU不代表逐像素不变。固定所选模型的2000次ROI配对bootstrap影增益95%区间[0.50546,0.88426]，不含模型选择与seed不确定性。

零初始化与同批基座535图预测逐像素相同；batch4与先前batch1基线有<0.0001点浮点差，因此最终保护判定使用独立batch1结果。基座状态哈希、分支更新、完整训练代码SHA、逐图混淆与转移矩阵独立复算通过。完整网络及轻量分支权重、曲线、诊断、ROI区间和摘要保存在outputs/phase445/phase445_20261106/network_run；远端D盘另保留535张新预测。完成一套新网络，单seed小幅正收益，不声称已实现70%或已证明具体分支机制。

## Phase446 双判别头局部遮光网络完整结果

沿用445环形地表参照，将单修正头拆为独立补漏/纠错头，按冻结教师非影/影预测路由；两头有符号输出、零初始化。新增0.1倍按路由与真实影/非影均衡的二分类损失。原431 best32k及BN全部冻结，目标标签监督0%；新增37691参数。8490源训练图、535原始源验证图、batch4、seed44501、6000步、峰值2e-4，预算与445相同。训练含7轮完整验证440.13秒。基座状态哈希、初始预测一致、所有新增头更新均通过。

最佳云影1000步，独立batch1复测地表88.870175、云84.450703、影62.048109、mIoU78.456329。相对431三类分别−0.257230、−0.292999、−2.139119点；相对445分别−0.284954、−0.434203、−2.840988点。6000步影61.322629（batch4）。无非影保护通过的训练检查点，70%目标失败。最佳点找回旧漏影613320，同时损伤原正确影715058，净新增漏影101738；消除旧误报739408，同时新增误报1081312，净新增误报341904。ROI固定模型影变化95%区间[−2.61504,−1.73772]。同一验证集选模型及复测，单seed。

保留445为当前更优方案；本轮同时改结构和损失，不能单独归因。所有轻量权重、曲线、逐图混淆/转移、ROI抽样及本地独立复算验收已存outputs/phase446/phase446_20261106/network_run；整网权重另存远端D盘。摘要READOUT.md，代码src/tools/*shadow446.py。

## Phase447 双头与均衡损失消融

A沿用446双头、去掉新增均衡损失；B沿用445单头、加入446的0.1倍路由内类别均衡二分类损失。两组并行各6000步，batch4、seed44501、峰值2e-4；原431 best32k及全部BN冻结，8490张原始源图训练、535张原始源验证图，0%目标标签监督。结构代码逐字对照、零初始化逐像素预测、基座状态哈希、新分支参数更新均验收通过。每1000步全验证，最佳点单张独立重载复测，逐图混淆/转移本地重新汇总与SHA验收通过。

A最佳1000步：地表89.004498、云84.648205、影63.598537、mIoU79.083746；影比431−0.588691，无保护通过检查点。B最佳及最佳保护均6000步：地表89.139801、云84.937556、影64.897589、mIoU79.658316；比431三类+0.012396/+0.193855/+0.710361点，保护通过，70%失败。B比445影仅+0.008493点，地表−0.015328、云+0.052650，基本持平。

相同6000步batch4的四格影IoU：445单头原损失64.889044；446双头均衡损失61.322629；A双头原损失63.332812；B单头均衡损失64.897781。双头在原损失下−1.556231点，在均衡损失下−3.575152点；均衡损失单头+0.008737、双头−2.010183，交互−2.018920点。支持淘汰本轮按原预测硬路由双头方案，保留简单445，B作为持平对照。单seed同验证集选择结果，不能推广为所有双头结构必然失败。

结果表、错误转移、代码签名与本地验收见outputs/phase447/phase447_20261106/RESULT.md及MAIN.csv。A/B目录保留轻量权重、完整曲线、逐图结果和ROI抽样，整网及预测图另存远端D盘。

## Phase448 分方向地表参照

保留445单头和原损失，将三个环形参照替换为上/下/左/右×三距离的12个局部窗口；1/4网格中心距离3/6/12、窗口3/5/9。冻结教师地表概率+0.1软加权，再按2×内容余弦相似度−4×平均色度差+log有效权重选择参照，越界候选屏蔽。无新增参数，总新增分支仍37002参数；原431及BN冻结，8490源训练、535原始源验证、0%目标监督，batch4、seed44501、6000步、峰值2e-4。方向/边界单元检查、初始预测一致、基座状态不变、新分支参数更新均通过。训练含7轮验证415.67秒，最佳影及最佳保护均6000步。

独立batch1重载复测地表89.159207、云84.887946、影64.907340、mIoU79.651498；相对431三类+0.031803/+0.144244/+0.720112点，保护通过、70%失败。相对445三类+0.004078/+0.003040/+0.018244点，mIoU+0.008454点。找回漏影263508、新增漏影122740，净找回140768；消除误报224882、新增误报278963，净新增误报54081。相比445影TP少1893、FP少7058，小幅收益来自减少误报。

107 ROI配对2000次bootstrap，固定所选448相对445的影增益95%区间[0.002630,0.034299]点，mIoU区间[0.002081,0.014952]点；单seed、同验证集选模型，未含选择不确定性。数值小幅正收益、距70%仍差5.092660点，保留候选但不足以支持持续参照选择微调。

完整曲线、权重、逐图混淆与错误转移、ROI抽样、跨版本配对区间及本地复算验收存outputs/phase448/phase448_20261106/network_run；摘要READOUT.md，网络src/tools/shadow_branch448.py。整网和535图预测另存远端D盘。

## Phase449 边界内外遮光判别

保留445环形参照、单头及全部损失，新增边界比较分支：平滑冻结教师影概率的中心差分提供法向，原图±2/6/12px双线性采样对数RGB、色度、局部纹理、地表/影概率；各尺度两侧差与中心偏差合计54通道，再加梯度幅度，共55通道；55→16逐点卷积、16通道深度3×3、16→1逐点卷积，末端零初始化，输出加入原有符号修正再限幅/门控。新增1073参数，总可训练38075。用fork_rng保持公共参数初始化与后续随机序列相同，单元验收通过；无标签边界输入。

原431及BN冻结，0%目标标签监督；8490原始源训练、535原始源验证；batch4、seed44501、6000步、峰值2e-4，与445损失相同。训练含7轮完整验证438.27秒，最佳影与最佳保护均6000步；独立batch1重载复测、基座哈希、初始预测一致、参数更新和本地逐图混淆汇总通过。

地表89.230663、云84.946959、影65.343557、mIoU79.840393。相对431三类+0.103259/+0.203258/+1.156329点；相对445三类+0.075534/+0.062053/+0.454461点、mIoU+0.197349；相对448影+0.436217。非影保护通过、70%失败（差4.656443）。相对原431净找回145916影像素、净减少36358误报；相对445总漏检−3255、误报−97497。

零新增前向的几何边界诊断：相对445，边界0—3px漏影+9047、误报−105528；3—8px漏影+8741、误报−1786；>8px漏影−21043、误报+9817。主要收益来自边界误报收回，并非边界召回提升。107ROI配对2000次bootstrap，固定所选449对445影增益95%区间[0.299674,0.600428]，mIoU[0.135855,0.254162]。单seed、同验证集选点。

保留449为当前较优网络，后续关注保持边界误报收益时提升云影召回。结果、所有轻量权重、逐图混淆/错误转移、边界分布、ROI区间及本地验收位于outputs/phase449/phase449_20261106/network_run/READOUT.md；整网和预测另存远端D盘。代码src/tools/shadow_branch449.py。

## Phase450 图像驱动遮光候选

完整保留449，新增不依赖教师预测边界的图像方向分支：原RGB按0.2126/0.7152/0.0722构造亮度，取log(亮度+0.02)后5×5平滑，负梯度提供指向暗部的法向；±2/6/12px采样并比较对数RGB、色度、纹理、地表/影概率，55通道经轻量卷积输出有符号修正，与449合并。新增1073参数，总可训练39148；没有将RGB解释为物理反射率。零输出初始化，fork_rng保持共享参数与随机序列一致，专项测试在教师均匀概率下能读出图像边界。

原431及BN冻结，0%目标监督；原始源训练8490、验证535，batch4、seed44501、6000步、峰值2e-4、损失完全沿用449，从头训练所有新增分支。训练含7轮完整验证492.68秒，最佳影和最佳保护均6000步。独立batch1重载复测、初始预测一致、冻结状态与参数更新验收、文件SHA及本地混淆矩阵复算全部通过。

地表89.251541、云84.971396、影65.559291、mIoU79.927409；相对431三类+0.124136/+0.227694/+1.372063点；相对449三类+0.020878/+0.024437/+0.215734点、mIoU+0.087016。保护通过，70%失败（差4.440709点）。召回77.475384→77.685686，非影误报率1.792011→1.785315。

相对449净少漏影25962、净少误报8565。边界0—3px漏检−40161、误报−9901；3—8px漏检+9964、误报−596；>8px漏检+4235、误报+1932。收益主要是边缘补漏，深部漏检仍未改善。固定所选模型107ROI配对2000次bootstrap影增益95%区间[0.160362,0.273198]、mIoU[0.064195,0.111292]点；单seed、同验证集选择。保留450当前候选，后续聚焦深部漏检。

结果、权重、逐图预测汇总、边界错误和ROI配对区间见outputs/phase450/phase450_20261106/network_run/READOUT.md；整网与逐图预测保存在远端D盘。网络src/tools/shadow_branch450.py。

## Phase451 远距同类地表参照

完整保留450，新增远距参照：1/4网格查询16×16全图候选，候选特征/RGB由教师地表概率+0.1加权池化；匹配分数4×内容余弦+2×色度余弦+log有效地表权重，屏蔽中心距离<8个1/4网格像素（原图32px）的候选。82通道内容及颜色差、参考权重和原概率经轻量卷积输出有符号修正。新增1505参数，总可训练40653；零初始化，fork_rng保留公共初始化和后续随机序列。测试通过均匀输入、跨出模拟暗区、梯度有限、初始化一致。

原431及BN冻结，0%目标标签监督；8490原始源训练、535源验证；6000步、batch4、seed44501、峰值2e-4，损失与450相同，分支从头训练。训练含验证532.26秒，最佳影和最佳保护均4000步；终点6000影65.514838（batch4）。独立重载batch1复测最佳点地表89.242874、云84.984682、影65.533166、mIoU79.920240；相对450三类−0.008667/+0.013286/−0.026125点，mIoU−0.007169；原431非影保护通过，70%未达到。

相对450漏检+19047、误报−23233。0—3px漏影+3676、误报−174；3—8px漏影−1015、误报+2829；>8px漏影+16386、误报−25888。召回77.685686→77.531398，本轮深部补漏目标失败。固定所选检查点107ROI配对2000次bootstrap影变化95%区间[−0.107539,+0.056118]、mIoU[−0.040497,+0.025855]，未显示明确增益，单seed同验证集选点。

保留450，451作为负结果存档。下一步可比较开放适配器/分割头与继续冻结，以检验训练范围是否限制深部修复。完整结果、权重、逐图混淆、边界分布、ROI区间与本地复算验收在outputs/phase451/phase451_20261106/network_run/READOUT.md；整网与预测另存远端D盘。代码src/tools/shadow_branch451.py。

## Phase452 同起点训练范围对照

两组均从450 step6000的原431+遮光分支权重开始，各续训6000步。A只训练39148参数新增分支；B额外开放1849672参数（空间CNN、交互、融合细化、局部读写、输出残差、分割头），DINOv3和全部BN统计及仿射参数冻结。基座eval但B启用梯度，梯度穿过冻结ViT回传到前置适配器。原431另作为完全冻结的难例、KL和softIoU参考，避免保护目标漂移。原始源训练8490、验证535、0%目标监督；batch4、seed45201、分支峰值5e-5、任务模块2e-6、200步预热后衰减，分组梯度裁剪。两组INPUT_TRACE全部相同（图像、标签、随机增强）；所有开放组实际更新，编码器/BN/参考哈希不变。并行训练含验证A854.18秒、B1608.24秒，不作独占资源吞吐比较。

A最高云影1000步独立复测65.601166%，地表89.248741低于450约0.002800点；严格保护最佳2000步，地表89.253604、云84.982736、影65.574548、mIoU79.936963，影仅+0.015257点，近似持平。B最高训练云影3000步，地表89.399334、云85.196261、影65.464513、mIoU80.020036；相对450三类+0.147793/+0.224865/−0.094778点，mIoU+0.092627。B各训练点非影保护通过但影均未超起点，故含step0的最佳保护选择回退450，独立复测step0与450完全一致。6000步终点batch4：A影65.562626、B影65.250486。

B最佳训练点相对450：漏检+6621、误报+11065；0—3px漏检−2303、误报−47342；3—8px漏检+6244、误报−7274；>8px漏检+2680、误报+65681。深部修复未达目标。A保护点对450固定模型107ROI配对2000次bootstrap影差95%区间[−0.015835,+0.048156]；B最佳训练点[−0.343442,+0.153567]。同验证集选点、单seed，未包含训练与选择不确定性。

保留450云影优先候选，A为持平对照，B记录为更高mIoU但较低云影IoU取舍。本轮不能支持简单扩大可训练范围即可解决深部瓶颈。完整重载535图、逐图混淆及错误转移本地复算、批次哈希、传输和权重SHA验收通过。结果outputs/phase452/phase452_20261106/RESULT.md；本地A1000/2000和B3000权重包含新增分支及student_patch，远端保留全部检查点与整网。训练src/tools/train_shadow452.py，评测src/tools/verify_shadow452.py。

## Phase453 遮光配对监督（2026-11-06）

- 沿用450网络与权重，冻结原431和BN，目标监督0%；源训练地表内合成20%–45%图像空间遮光，核心标影、过渡忽略。原图损失+0.5配对损失，6000步、batch4、lr5e-5、seed45201；额外配对前向，不是等算力对照。
- 独立535张最佳训练点1000步：地表89.0952、云84.9618、云影65.1002、mIoU79.7190；相对450云影-0.4591点。6000步终点云影64.3378（batch4）。全部训练点未保住450地表IoU，保护选择回退step0。
- 净找回231258影像素、深部漏检减少64756，但新增误报458397（地表303266、云155131）。召回77.6857→79.5590%，非影误报率1.7853→2.1437%。保留450，本轮负收益。
- 原始批次签名与452A一致；冻结、增强标签保护、独立重载及本地535张混淆矩阵验收通过。结果：`outputs/phase453/phase453_20261106/RESULT.md`，代码：`src/tools/train_shadow453.py`、`augment_shadow453.py`。

## Phase454 真实云影/困难非云影排序（2026-11-06）

- 从450最佳分支续训6000步，原431/BN冻结、目标监督0%；仅真实源训练图。450损失增加0.1倍图内困难像素排序，每图地表/云各最多1024对，margin1。网络结构不变，batch4、lr5e-5、seed45201，与452A原图批次签名一致。
- 独立535张最佳1000步：地表89.1985、云84.9159、影64.1227、mIoU79.4124；影比450低1.4366点。6000步终点影62.9018（batch4）。所有训练点不满足保护450两类的门槛，回退450。
- 误报减少401354、漏检增加467509（深部+25380）。召回77.6857→73.8987%，非影误报率1.7853→1.4715%。
- 零训练4096桶得分诊断（无截断）：450/454近似像素AP87.6671/87.2801%；匹配450原误报率，召回77.6857/76.4291%。不只是默认阈值偏移，匹配误报后也未恢复450表现；不用于选阈值报主结果。
- 损失梯度与ignore屏蔽、冻结、独立重载、本地混淆矩阵及权重SHA验收通过。结果：`outputs/phase454/phase454_20261106/RESULT.md`；代码：`src/tools/train_shadow454.py`、`loss_shadow454.py`、`operating_shadow454.py`。

## Phase455 地表—遮光一致性模块（2026-11-06）

- 450增加2594参数，整分支41742参数：1/4网格8方向×3距离的亮度比例、色彩差、纹理相关、局部仿射补偿残差与语义参照选择，融合云软上下文；末层零初始化并入原修正量。保持450损失、固定argmax，不搜索阈值。RGB一致性为线索而非物理标定证明。
- 源训练8490、验证535，目标监督0%，原431和BN冻结；6000步、batch4、seed45201，旧lr5e-5、新2e-4。初始化精确复现、全部新增参数更新、冻结与原图批次签名验收通过。
- 独立最佳影1000步：地表89.2493、云84.9795、影65.6015、mIoU79.9434；保护点2000步：89.2568/84.9845/65.5673，m79.9362。保护点影仅比450高0.0080点，CI[-0.0306,0.0475]。普通续训452A最佳/保护影65.6012/65.5745，无实质超越。
- 同权重关闭新模块影65.5984，开启65.6015，仅+0.0030点；不是独立训练消融。最佳影点净漏检-22939（深部-7618）、误报+25559。保留450，不作为有效结构提升。
- 结果：`outputs/phase455/phase455_20261106/RESULT.md`；新模块`src/tools/continuity455.py`，网络`shadow_branch455.py`，训练`train_shadow455.py`，整网推理`infer_shadow455.py`。

## Phase456 云—影位移关联层（2026-11-06）

- 从450出发增加9441参数（全分支48589）：8方向×4距离移位云概率，归一化云/影空间匹配与场景特征共同学习32位移+空关联权重；方向投影、距离、有效覆盖、空关联写入像素头/门控前的32通道特征。末层零初始化，固定argmax，不调阈值。
- 原431/BN冻结，目标监督0%，源8490图训练6000步、batch4、seed45201，旧lr5e-5、新2e-4，沿用450损失。零初始化、实际参数更新、冻结和与452A原图批次签名验收通过。
- 独立535图：最高影1000步89.2487/84.9796/65.6016，m79.9433；保护点2000步89.2545/84.9841/65.5674，m79.9354。保护影仅+0.0081点，CI[-0.0266,0.0453]。未超过普通续训452A的实用水平，保留450。
- 同权重关/开新层云影65.5956/65.6016，仅+0.0060点。最佳影点净漏检-22564（深部-7455）、误报+24967。近幅度饱和像素仅0.0017%，新层shadow logit平均绝对变化约0.00351；不支持把低收益归因于普遍幅度饱和。标签辅助可行性上界仅作诊断，不是预测性能。
- 结果`outputs/phase456/phase456_20261106/RESULT.md`；模块`src/tools/cloud_geometry456.py`，网络`shadow_branch456.py`、训练`train_shadow456.py`、推理`infer_shadow456.py`。

## Phase457 弱通路诊断与去局部捷径（2026-11-06）

- 固定64张源训练图、零更新诊断456：6000步注入RMS0.108505，云位置干预变化0.004028；保护梯度未压倒CE。直接读取旧局部特征是一条可能捷径，干预不足以单独证明根因。
- 新结构仅从13通道云—影关联证据生成空间中心化缩放/偏移，调制归一化旧特征，去掉注入器的局部特征/概率直连。新增9313参数，全分支48461。450起点、原431/BN冻结、目标监督0%，6000步、batch4、seed45201，旧lr5e-5、新2e-4，原损失、固定argmax。
- 独立535图：最佳影1000步89.2487/84.9789/65.5995，m79.9424；保护2000步89.2532/84.9814/65.5657，m79.9334。保护影仅+0.0064点，CI[-0.0245,0.0403]，不优于普通续训。
- 改造后6000步注入RMS0.029751、云位置干预0.031490，绝对干预响应约7.82倍；同样本、零更新签名检查通过。模块开启/关闭影65.5995/65.6007，无实质增益，保留450；依赖增强不等于有用信息增强。
- 结果`outputs/phase457/phase457_20261106/RESULT.md`；诊断`src/tools/diagnose_shadow457.py`、`diagnose_after457.py`；新模块`cloud_geometry457.py`、训练`train_shadow457.py`。

## Phase458 有监督云影区域传播（2026-11-06）

- 新层4747参数（整分支43895）：云影种子、语义/色彩/亮度邻接连接、1/2/4/8步长保留种子的传播，写回像素头前特征。450损失加0.05种子BCE、0.05传播结果BCE、0.05连接BCE；连接监督仅纯度>=90%有效源patch。
- 450起点、原431/BN冻结，目标监督0%；6000步、batch4、seed45201、旧lr5e-5/新2e-4、固定argmax。初始化复现、所有新参数非零梯度、ignore屏蔽、冻结和独立535图验收通过。
- 最高影1000步89.2497/84.9787/65.6014，m79.9433；影+0.0421点但地表-0.0018点，与普通续训452A影65.6012几乎相同。预定完整模型保护选择回退step0；6000步影65.5334，保留450。
- 同权重完整/整层关闭/仅传播关闭影65.601431/65.585965/65.601425，传播差约0.000006点；整层关闭事后对照地表89.2553、云84.9807，不计为预定新网络收益。净漏检-21781（深部-7279）、误报+23805。当前冻结基座残差路径下无实质结构增益。
- 结果`outputs/phase458/phase458_20261106/RESULT.md`；模块`src/tools/shadow_region458.py`，训练`train_shadow458.py`，三路开关`ablate_shadow458.py`。

## Phase459 云内容／遮光分解与解码器联合训练（2026-11-06）

- 参考SDDNet的背景/阴影分解思想，新增263432参数：四尺度内容/遮光双路、软交互、零初始化特征注入到Mask2Former像素解码器；云/地表、非云区阴影、特征重建辅助监督。未宣称物理解耦。
- 两臂450起点、源8490/验证535、目标监督0%、固定argmax；A原450+解码器续训，B加入新层和辅助损失。共同6000步、batch4、seed45201、解码器lr1e-5、分支5e-5；VFM/空间适配器/BN固定。
- 独立batch1最佳训练点：A3000步89.3982/85.1897/65.3340，m79.9740；B1000步89.3845/85.1265/65.3001，m79.9370。B比450影-0.2592点，地表+0.1330、云+0.1552；两臂保护最优均回退step0，保留450。
- 同权重B关/开新路径影65.0251/65.3001，开启+0.2749点，107ROI区间[-0.3424,0.9244]。B-A各自最佳点影-0.0340，区间[-0.3880,0.3382]；同6000步B-A+0.0513点，仍低于450。新路径有作用，但无整体净收益。
- 初始端点、梯度/ignore标签、全部64个新增张量更新、冻结状态、两臂批次SHA、535图独立复算及本地权重验收通过。单seed探索；6000步续训，不是40k全量重训。LOCK的骨干梯度描述为继承文字，实际全部backbone冻结。
- 结果`outputs/phase459/phase459_20261106/RESULT.md`；模块`src/tools/shadow_decomposition459.py`，训练`train_shadow459.py`，推理`infer_shadow459.py`，消融`ablate_shadow459.py`。

## Phase460 无遮挡地表参照解码器（2026-11-06）

- 借鉴CANet的非阴影上下文传递，新增196876参数：四尺度软区域预测、全图地表参照注意力、自身/参照差异与云上下文融合，注入像素解码器；0.1区域软标签CE辅助监督。推理不输入标签，不设无云无影硬规则。
- 450起点、源8490/原始验证535、目标监督0%、固定argmax；6000步、batch4、seed45201，解码器lr1e-5、分支5e-5，全部骨干/BN固定。复用相同设置459A普通续训对照，输入批次及协议签名一致。训练用时954.35秒。
- 独立最佳1000步地表89.3892、云85.1304、影65.3778、m79.9658；相对450分别+0.1376/+0.1590/-0.1815点。比459A最佳影+0.0438点，107ROI区间[-0.3619,0.4608]。保护最优回退step0，保留450。
- 同权重完整/整层关/仅参照关：云影65.3778/65.1187/65.3753。整层作用+0.2592点，但参照读取仅+0.0025点，区间[-0.0062,0.0116]；无实质参照机制收益。相对450漏检+30385、误报-5869。
- 零初始化、全部52个新增参数张量更新、冻结状态、独立535图重算、两种开关消融、批次一致性与本地权重签名验收通过。6000步续训、单seed探索，不是全量40k重训。
- 结果`outputs/phase460/phase460_20261106/RESULT.md`；模块`src/tools/shadow_reference460.py`，训练`train_shadow460.py`，整网推理`infer_shadow460.py`，参照消融`ablate_reference460.py`。

## Phase461 稀疏地表对应与遮光差异（2026-11-06）

- 460零更新诊断：固定64张源训练图，四尺度归一化注意力熵0.9424～0.9671、top8质量5.8%～8.2%，参照分散；不单独当作根因证明。
- 新增166156参数：top8语义/色度匹配地表参照；仅用语义差、观测RGB对数差、色度差、相对纹理差、置信度与余弦差注入像素解码器，去掉直接局部特征输入。观测RGB差不解释为物理透射率。
- 450起点，源8490/原始验证535，目标监督0%，固定argmax。6000步、batch4、seed45201；解码器lr1e-5、分支5e-5，骨干/BN固定；对照复用459A和460。训练957.45秒。
- 独立最佳1000步地表89.3785、云85.1308、影65.3607、m79.9567；相对450+0.1269/+0.1594/-0.1986点。较459A最佳影+0.0266点，107ROI区间[-0.3711,0.4404]；较460影-0.0171点。保护选择回退step0，保留450。
- 同权重完整/整层关/参照关/参照错位影65.3607/65.1549/65.3629/65.3623。参照开启-0.0023点，区间[-0.0121,0.0049]；正确对应相对错位-0.0016点，区间[-0.0081,0.0042]。错位为查询网格展平后固定循环移位17位置，非完全随机置乱。无实质参照机制收益。
- 全部52个新参数张量更新、初始端点、冻结状态、协议/批次签名、独立535图重算和本地权重SHA验收通过。6000步续训、单seed探索，未达到70%目标。
- 结果`outputs/phase461/phase461_20261106/RESULT.md`；诊断`src/tools/diagnose_shadow461.py`，模块`shadow_sparse461.py`，训练`train_shadow461.py`，位置错位干预`ablate_shuffle461.py`。

## Phase462 区域查询监督与云影内部/边缘分解（2026-11-06）

- 训练方法实验，无新增网络参数。A增加有效区域加权的匈牙利查询匹配、分类CE、掩膜BCE/Dice，权重0.1；B把云影内部和4像素边缘作为同类双区域。不是原431随机点损失的逐项复现，几何边缘不等于物理半影。
- 450起点，源8490/原始验证535，目标监督0%，固定argmax；两臂各6000步并行，batch4、seed45201，解码器lr1e-5、分支5e-5，骨干/BN固定。耗时1763/1764秒。
- 独立最佳A1000步：地表89.3955、云85.1964、影65.3694、m79.9871；B3000步：89.3706/85.1668/64.8244，m79.7873。相对450影-0.1898/-0.7349点，保护选择均回退step0，未达到70%，保留450。
- A比普通续训459A影+0.0354点，107ROI区间[-0.2096,0.2900]；B比A影-0.5451点，区间[-0.9841,-0.1267]，所有相同步数B均更差。当前设置不支持边缘拆分。
- A少漏23987但多误报79178像素；B多漏53914、多误报82679。B距边界8像素以内误报少109068，但远处误报多191747，近边界漏检多61458。
- 标签守恒、有限梯度、ignore零梯度、全忽略零损失、冻结状态、协议/抽查批次签名、独立535图复算及最佳权重SHA验收通过。单seed、验证集选点探索。
- 结果`outputs/phase462/phase462_20261106/RESULT.md`；损失`src/tools/query_loss462.py`，训练`train_shadow462.py`，独立评测`verify_shadow462.py`。

## Phase463 方向遮光线索前置空间分支（2026-11-06～07）

- 新增17443参数：1/4网格左右上下、三距离RGB对数共同变化/色差，与原空间细节融合；零初始化注入context/detail，进入四次交互。参考CVPR2018方向上下文和WACV2025材质一致边界，未声称物理光照不变或复现原文。
- 450起点，源8490/原始验证535，目标监督0%，固定argmax；6000步、batch4、seed45201，分支lr5e-5、解码器1e-5，0.1区域软标签CE。原骨干/BN固定，梯度经过冻结ViT；训练1091.53秒。
- 独立最佳3000步：地表89.3794、云85.1491、影65.4102、m79.9796；相对450+0.1278/+0.1777/-0.1491点，保护选择回退step0，保留450，未达到70%。普通续训459A影65.3340，新结构+0.0762点，107ROI区间[0.0003,0.1855]。
- 同权重完整/整层关/全部RGB线索关/仅方向差异关：影65.4102/65.3978/65.4133/65.4136。整层作用+0.0124点，区间[-0.0810,0.1285]；方向差异开启-0.0034点，区间[-0.0087,0.0014]。无实质方向线索收益。
- 相对450漏检+17939、误报+5920；距边界8像素内误报-80713，远处误报+86633。新增14个张量全部更新，初始端点、梯度/ignore标签、冻结状态、独立535图及三种消融、批次/权重SHA验收通过。
- 单seed6000步续训；停止重复叠加类似RGB差异支路。结果`outputs/phase463/phase463_20261106/RESULT.md`；模块`src/tools/shadow_spatial463.py`，训练`train_shadow463.py`，方向消融`ablate_direction463.py`。

## Phase464 交互内部的云/地表/云影路由（2026-11-07）

- A解冻原四次交互与解码器；B相同解冻范围新增5688参数：query/key软三类预测、每头3×3关系矩阵，以4*tanh关系形成注意力logit偏置，零初始化。四层query/key平均0.1软标签CE，ignore覆盖加权。B-A包含路由与辅助监督联合训练效应。
- 450起点，源8490/原始验证535，目标监督0%，固定argmax；两组各6000步，batch4、seed45201，交互/解码器lr1e-5，分支5e-5。VFM及其他原骨干模块/BN固定。用时A1943.44秒、B1964.26秒。
- 独立最佳均1000步：A地表89.3392、云85.1632、影65.2705、m79.9243；B89.3386/85.1408/65.3036，m79.9277。相对450影-0.2888/-0.2557点，保护选择均step0，保留450，未达到70%。
- B-A影+0.0331点，107ROI区间[-0.0046,0.0832]；2000～6000步B均低于A。同权重路由关/开影65.303710/65.303610，开启-0.000100点，区间[-0.001620,0.001358]，无实质路由推理收益。
- A相对450漏检+11295、误报+47424；B漏检+11006、误报+40421。原交互32张量、新路由20张量实际更新，其余骨干/BN固定，初始端点、梯度、ignore标签、独立535图/消融、批次签名及权重SHA验收通过。
- 单seed6000步续训，不是40k全量重训。结果`outputs/phase464/phase464_20261107/RESULT.md`；模块`src/tools/shadow_routing464.py`，训练`train_shadow464.py`，消融`ablate_shadow464.py`。

## Phase465 尺度控制与分区上下文读取（2026-11-07）

- 零更新诊断464：固定32张源训练图，第四交互原注意力logit行std28.42/路由偏置std0.00106；路由幅度不足。分割/保护损失路由梯度平均余弦为正，未见平均方向明显冲突。数学SDPA用于重复求梯度，完整签名/抽样见ROUTING_DIAG.json。
- A新增尺度控制读取通道：注意力分数去均值，std>1时缩放到1。B再按key软三类概率分区归一化读取value，query类别及每头3×3关系矩阵混合。新增5688参数，A不使用的关系矩阵固定；两组同0.1区域CE。新通道系数前1000步0→0.5，之后固定，避免弱门控自行退出。
- 450起点，源8490/原始验证535，目标监督0%，固定argmax；两组各6000步、batch4、seed45201，交互/解码器lr1e-5，分支5e-5，其余原骨干/BN固定。用时A2400.48秒、B2594.75秒。
- 独立最佳均1000步：A89.3621/85.1804/65.3232，m79.9552；B89.3597/85.1744/65.3083，m79.9475。相对450影-0.2361/-0.2510点，保护最佳均step0，保留450，未达到70%。B-A影-0.0149点，107ROI区间[-0.0494,0.0162]。
- B完整/新读取全关/仅类别分区关影65.308336/65.256126/65.308018。整条读取开启+0.052209点，区间[-0.024836,0.177496]；类别开启+0.000318点，区间[-0.026639,0.025061]，无明确类别机制贡献。B相对450漏检+16713、误报+30621。
- 补充450原四次交叉注意力全关：89.1967/84.7931/65.5698，m79.8532；影净+0.0105点，但召回-0.9676点、地表/云-0.0549/-0.1783点；漏检+119452、误报-184528。不能替换450；当前权重下交互对云影IoU净贡献小，不代表无训练价值或无错误构成作用。
- A新增16预测器张量更新、4未使用关系矩阵固定；B20张量全部更新，原交互更新、其余骨干/BN固定，初始端点、固定0.5增益、独立535图/消融、协议批次/权重SHA验收通过。单seed6000步续训，非40k全量重训。
- 降低交叉注意力路由小改优先级，后续先核查MsRE/局部读写贡献。结果`outputs/phase465/phase465_20261107/RESULT.md`；模块`src/tools/shadow_routing465.py`，诊断`diagnose_routing465.py`，原交互消融`ablate_original465.py`。

## Phase466（2026-11-07）：固定450权重的MsRE与局部读写诊断

- 源域CloudSEN535张、目标监督0%、零训练、固定BN和argmax；6种干预3210次前向，107个ROI配对bootstrap及状态哈希验收通过。
- 基准云影65.5593%；关闭MsRE40.8032%，关闭局部读取65.5304%，关闭局部反馈65.5610%，同时关闭读写65.5312%。
- MsRE内部：关闭多尺度token读取65.3923%，关闭1×1投影43.8379%。当前权重明显依赖投影路径，token与读写的云影净影响较小；不同干预损失不能相加，也不代表重新训练时的贡献。
- 本轮未训练新网络，最优仍为450。下一步优先在MsRE投影主路径做遮光对比结构，并设同投影继续训练对照。
- 完整结果：`../outputs/phase466/phase466_20261107/RESULT.md`；计划：`research_plans/phase466.md`。

## Phase467（2026-11-07）：MsRE投影内局部—环形遮光对比

- 目标监督0%，CloudSEN训练8490/验证535；450起点，A/B各6000步，固定BN与argmax。A只训练原MsRE投影；B新增21712参数环形对比分支，四阶段共享；原解码器、450分支和其余骨干固定。
- 独立最佳：A地表/云/影89.250732/84.981573/65.535307；B89.255897/85.007388/65.522634。B影较450低0.036656点，B-A -0.012673点，107ROI区间[-0.094410,0.066829]。
- B新增分支on-off影-0.010693点；仅环形对比on-off +0.001229点，均无明确正收益。B漏检净减68789像素、误报净增113169像素；保护规则最优仍为初始450。
- 起点逐张量与431/450原权重核对、原投影及B全部新参数更新、其他参数固定、两组批次一致、535独立混淆矩阵、SHA及双端归档验收通过。单seed探索性续训，未达70%。
- 结果：`../outputs/phase467/phase467_20261107/RESULT.md`；结构：`tools/shadow_projection467.py`；计划：`research_plans/phase467.md`。

## Phase468（2026-11-07）：高分辨率亮度—色度分频解码

- 目标监督0%，CloudSEN训练8490/验证535，450起点；A普通细节桥23128参数，B分频桥25657参数；仅新增桥与解码器训练，原骨干、450分支、BN固定。两组各6000步，固定argmax。
- 独立最佳均3000步：A地表/云/影89.414014/85.227906/65.331109，mIoU79.991009；B89.417702/85.236376/65.336380，mIoU79.996819。B比450 mIoU高0.069410点，但影低0.222911点，未达70%，保留450。
- B-A影+0.005271点，107ROI区间[-0.009888,0.018788]。同权重整桥on-off影-0.069008点；频带信息on-zero +0.014321点，区间[-0.008883,0.039403]。后者保留全部参数/偏置。
- 分频信息降低0–3像素边界漏检15686，同时增加8像素外误报13445；局部边缘作用未转化为全图云影净收益。
- 初始源权重逐张量核验、实际共享初始化/训练批次一致、新参数及解码器更新、原参数冻结、535图独立混淆矩阵、SHA及双端归档验收通过。单seed探索性续训。
- 结果：`../outputs/phase468/phase468_20261107/RESULT.md`；结构：`tools/shadow_detail468.py`；计划：`research_plans/phase468.md`。

## Phase469（2026-11-07）：VFM内部云遮挡／地表遮光双路适配

- 目标监督0%；CloudSEN训练8490/验证535，450起点，A/B各6000步、batch4、seed45201、固定argmax。DINOv3末4层QKV加入rank4云路与影路，三类token软门控；A加性影V残差，B原V乘有界相对修正。各新增41484参数，原网络全部权重及BN冻结。
- 独立最佳均1000步：A地表/云/影89.275916/85.027243/65.358219，mIoU79.887126；B89.276164/85.026306/65.350106，mIoU79.884192。云影分别较450低0.201071、0.209185点，未达70%，保留450。
- B-A影-0.008114点，107ROI区间[-0.018752,0.001677]；同权重关闭全部新增层精确恢复450，相对调制改加性仅差-0.000470点，区间跨0。
- B漏检净减78166、误报净增166437，其中距离影边界8像素外误报增加120371；召回改善未转化为云影净收益。
- 新增24个参数张量全部更新、原权重冻结、实际共享初始化与训练批次一致、535图独立混淆矩阵和传输SHA通过。训练各约29分钟；单seed6000步探索。
- 结果：`../outputs/phase469/phase469_20261107/RESULT.md`；结构：`tools/shadow_vfm469.py`；计划：`research_plans/phase469.md`。

## Phase470（2026-11-07）：漏检恢复—误报抑制竞争专家

- 目标监督0%；CloudSEN训练8490/验证535，450起点，A/B各6000步，batch4、seed45201、固定argmax。解码器前1/4特征加入4专家共44572参数，原网络全部冻结。A等权混合，B按通用/遗漏云影/地表误报/云误报软路由，标签由固定450预测与源训练标签生成。
- 独立最佳均1000步：A地表/云/影89.255524/84.987440/65.412604，mIoU79.885189；B89.258007/84.988997/65.348595，mIoU79.865200。云影较450分别低0.146686、0.210696点，保留450，未达70%。
- B-A影-0.064009点，107ROI区间[-0.102968,-0.026825]。B同权重改等权后影65.453750；竞争路由净作用-0.105155点，区间[-0.212573,-0.002837]。关闭整个新增模块精确恢复450。
- B少漏检49461、多误报122853像素，其中8像素边界外新增误报89671。路由对遗漏影/地表误报/云误报的对应平均权重为31.78%/43.03%/50.22%，有类型区分但未转化为纠错收益。
- 新增32个参数张量更新、原参数和BN固定、共享初始化/批次一致、535图独立混淆矩阵、同权重干预及路由诊断预测复现、SHA验收通过。两组训练各约27分钟。单seed探索。
- 结果：`../outputs/phase470/phase470_20261107/RESULT.md`；结构：`tools/shadow_experts470.py`；计划：`research_plans/phase470.md`。参考DSDNet的干扰区域学习思路，不将错误监督或多专家本身宣称为首次提出。

## Phase471（2026-11-07）：云影轮廓法向特征位移

- 目标监督0%；CloudSEN训练8490/原始验证535，450起点，A/B各6000步、batch4、seed45201、固定argmax。解码器前1/4特征新增16348参数的距离场与位移层；A自由二维位移，B法向投影，最大8原图像素。两组辅助距离监督相同，原网络全部冻结。
- 独立最佳均1000步：A地表/云/影89.248613/84.966734/65.559975，mIoU79.925107；B89.249184/84.967528/65.559122，mIoU79.925278。A影+0.000684点、B影-0.000168点，但两者地表/云略降，无有效收益，保留450。
- B-A影-0.000852点，107ROI区间[-0.002527,0.000801]；B法向on-off影-0.000805点，区间[-0.002190,0.000604]。关闭新增模块精确恢复450。
- B漏检净增3像素、误报净增33像素。全图平均预测位移0.107595像素，门加权位移代理量0.009598像素，特征修正RMS/原特征RMS逐图均值0.060149%；这些全图平均不能直接解释局部作用。辅助粗网格距离符号IoU51.2334%，不与主表混用。
- 新增18个参数张量更新、原权重和BN固定、共享初始化/训练批次一致、535图独立混淆矩阵、同权重干预及位移诊断复现、SHA验收通过。每组训练约26分钟，单seed探索。
- 结果：`../outputs/phase471/phase471_20261107/RESULT.md`；结构：`tools/shadow_transport471.py`；计划：`research_plans/phase471.md`。参考特征位移方法，本轮检验云影距离场和法向约束。

## Phase472（2026-11-07）：MsRE主路径邻域遮光对比投影

- 目标监督0%；CloudSEN训练8490/原始验证535，450起点，A/B各6000步、batch4、seed45201、固定argmax。A续训原MsRE projector与解码器（1464356参数）；B同范围加6147参数选择器，对8个间隔2-token邻居按语义/色度/较亮偏好选择软参照，原投影输入改为x+0.25*(x-reference)。
- 原模块lr1e-5、新模块1e-4；其余原参数、450分支、BN及VFM冻结。B强度前2000步由0升至0.25，均只在2000步及以后选训练最优点。两组无额外辅助损失。
- 独立最佳均3000步：A地表/云/影89.335016/85.112988/65.275841，mIoU79.907948；B89.344197/85.126399/65.233503，mIoU79.901366。云影较450分别低0.283450、0.325788点，保留450。
- B-A影-0.042338点，107ROI区间[-0.215567,0.127376]；B同权重对比on-off影-0.024528点，外观偏好on-off +0.001252点，区间均跨0。关闭新结构保留已训练原参数，不应期待恢复450。
- B少漏检14214、多误报94847像素；边界8像素外误报增加130124。四阶段投影输入修正RMS比例均值13.48%/16.61%/14.22%/4.67%，已明显参与主路径但无云影净收益。
- 原投影2张量、解码器102张量和B新增4张量实际更新；未开放参数与BN固定；共享原参数初始化、固定seed及记录批次签名、535图独立混淆矩阵、干预、预测复现和SHA验收通过。两组训练约32/34分钟，单seed6000步续训探索。
- 结果：`../outputs/phase472/phase472_20261107/RESULT.md`；结构：`tools/shadow_contrast472.py`；计划：`research_plans/phase472.md`。参考阴影检测对比信息动机，不声称RGB参照是真实无影反射率。

## Phase473：类别引导遮光参照（2026-11-07，完成）

- 目标：保护450地表/云IoU并提高云影，目标70%。源训练8490/原始验证535，目标监督0%，固定argmax；6000步，batch4，seed45201，RTX4090D，FP32、输入512²、warmup200。
- 在472对比投影上新增三类辅助头：云影概率控制查询修正，邻居地表概率引导参照；保留语义/色度/亮度匹配。四阶段共享6198参数，原MsRE投影+解码器1464356参数开放，其他原参数、450分支、VFM与BN冻结。原模块lr1e-5/新模块1e-4，类别软占比辅助CE权重0.1。
- 初始强度0精确复现450，2000步升至0.25；选2000步以后最优。复用472A同条件普通续训，已核验共同初始化及记录批次签名。此为结构与辅助监督整体方法比较。
- 最优3000步：地表/云/影89.337760/85.109195/65.301987，mIoU79.916314；相对450 +0.086219/+0.137799/-0.257304点，未通过，保留450。相对472A云影+0.026145，107ROI区间[-0.038299,0.104475]。
- 同权重整个模块on-off影+0.031517；类别路由on-off +0.081387；相对保留平均门控的图内均匀路由+0.022598；区间均跨0。类别概率有区分能力，但尚无明确结构收益。
- 较450多漏检5476、多误报49254；>8像素远边界误报+88837。四阶段输入修正RMS均值1.86%/2.93%/3.11%/0.93%。6000步末点云影65.192364。
- 全程训练约20.6分钟，峰值分配显存5960607744字节；该耗时与上一轮并行实验不作速度比较。完整535图独立评测、三项干预、诊断、更新/冻结与SHA通过；单seed探索。
- 结果：`../outputs/phase473/phase473_20261107/RESULT.md`；代码：`tools/shadow_contrast473.py`；计划：`research_plans/phase473.md`。

## Phase474：云内容—地表遮光条件解码（2026-11-07，完成）

- 动机：检验类别层级能否减少影分支对云的竞争，门槛保持450地表/云IoU且影70%。目标监督0%，CloudSEN训练8490/原始验证535，输入512²、FP32、RTX4090D，两组并行6000步、batch4、seed45201、lr1e-4、warmup200。
- 原450完整网络与BN全部冻结。两条独立CNN/语义投影，云内容路读取场景，地表遮光路读取三尺度局部亮度/色度/纹理。每组31106新参数。A普通三类残差组合，B云/非云与非云内影/地表条件组合。两组同初始化/损失/记录批次，无辅助标签损失，固定argmax。
- B的影分支不直接改变云概率；最终argmax及云IoU仍需实测。冻结参考同次前向缓存与单独推理完全一致。
- A最优1000步：地表/云/影89.254241/84.979557/65.557957，mIoU79.930585；相对450三类变化+0.002700/+0.008161/-0.001334点。漏检净变化-15704、误报净变化24252像素。训练12.11分钟，峰值分配显存1923955200字节。
- B最优1000步：地表/云/影89.249331/84.980837/65.522933，mIoU79.917700；相对450三类变化-0.002210/+0.009441/-0.036358点。漏检净变化-31521、误报净变化56224像素。训练12.16分钟，峰值分配显存1943915008字节。
- B−A云影-0.035024点，107ROI配对区间[-0.07087568937070259, -0.0014038439286391488]。
- B_decoder_on_minus_off：影-0.036358点，区间[-0.09399466082931732, 0.014592542371405405]。
- B_conditional_on_minus_off：影-0.054444点，区间[-0.09735036285473733, -0.011783665647444548]。
- B_spatial_minus_mean：影+0.007898点，区间[-0.004344905137177157, 0.022129163656313358]。
- 决策：未获得保护地表/云条件下的云影净提升，保留450。
- 独立535图评测、三项同权重干预、参数更新与冻结、初始化/记录批次/权重SHA通过。单seed验证集选点探索。
- 结果：`../outputs/phase474/phase474_20261107/RESULT.md`；代码：`tools/shadow_conditional474.py`；计划：`research_plans/phase474.md`。

## Phase475：云影误报的损失权重对照（2026-11-07，完成）

- 诊断训练目标，不是新架构。固定474A普通组合网络，原450与BN冻结，仅训练31106参数。目标监督0%，CloudSEN训练8490/验证535、512²、FP32、RTX4090D；新A/B各6000步、batch4、seed45201、模块seed47401、lr1e-4、warmup200，固定argmax。
- 旧474A：云影难例4、正确影1.5、其他1；新A：云影难例4、其他1；新B：全部1。其他KL、softIoU及正则相同。三组同初始权重与记录批次；对照单因素逐级移除，检验误报增长是否来自像素权重。
- A最优1000步：地表/云/影89.253828/84.977219/65.557313，mIoU79.929453；相对450三类变化+0.002287/+0.005823/-0.001978点。漏检净变化-7272、误报净变化11534像素。末点影IoU65.371359；训练12.16分钟，峰值分配显存1923955200字节。
- B最优1000步：地表/云/影89.256148/84.978401/65.553725，mIoU79.929424；相对450三类变化+0.004607/+0.007005/-0.005566点。漏检净变化8130、误报净变化-11160像素。末点影IoU65.479404；训练12.16分钟，峰值分配显存1923955200字节。
- B−A影-0.003588点，107ROI区间[-0.023972022575062014, 0.019418932791286276]。
- A_minus_474A：影-0.000644点，区间[-0.009895783169932493, 0.009774534348219874]。
- B_minus_474A：影-0.004232点，区间[-0.03369353225180642, 0.028891938453963897]。
- B_decoder_on_minus_off：影-0.005566点，区间[-0.017120938363241224, 0.007238003149560147]。
- B_spatial_minus_mean：影+0.002466点，区间[-0.004825052592634549, 0.009113534592411198]。
- 决策：未获得保护地表/云条件下的云影净提升，保留450。
- 独立535图、107ROI配对区间、B关闭新解码与空间均值干预、参数更新/冻结、三组初始化/记录批次及权重SHA通过。当前网络与单seed训练设置的诊断。
- 结果：`../outputs/phase475/phase475_20261107/RESULT.md`；训练：`tools/train_shadow475.py`；计划：`research_plans/phase475.md`。

## Phase476：材质引导空间卷积（2026-11-07，完成）

- 动机：通过亮度弱敏感的邻域引导跨越阴影明暗边界，检验云影/暗地物的特征区分能力。目标保护450地表/云且影70%。目标监督0%，CloudSEN训练8490/验证535、512²、FP32、RTX4090D。
- 在空间CNN1/4细节上加入8邻居逐通道差分卷积及1×1混合，并池化修正1/16上下文，位于全部VFM交互之前。A按logRGB距离引导，B按去通道均值log色度距离引导。各4609新参数，零差分初始化。色度只是材质线索，不等于物理反射率。
- 两组各6000步、batch4、seed45201/模块seed47601，新模块lr1e-3、原解码器1e-5、warmup200；其他原参数、450分支、BN冻结。统一不加权CE及已有保持约束，固定argmax。
- A最优1000步：地表/云/影89.451075/85.184702/65.652485，mIoU80.096088；相对450三类变化+0.199534/+0.213306/+0.093195点。漏检净变化10598、误报净变化-36908像素；末点影IoU65.571808。训练32.90分钟，峰值分配显存6059683328字节。
- B最优3000步：地表/云/影89.483830/85.243720/65.644845，mIoU80.124132；相对450三类变化+0.232289/+0.272324/+0.085555点。漏检净变化13734、误报净变化-39987像素；末点影IoU65.586624。训练33.09分钟，峰值分配显存6059683328字节。
- B−A影-0.007640点，107ROI区间[-0.2103417986072081, 0.19071274961830015]。
- A_convolution_on_minus_off：影-0.024094点，区间[-0.0853166569760603, 0.030213879680536935]。分别为空间卷积开关、色度对logRGB引导。
- B_contrast_on_minus_off：影-0.081530点，区间[-0.2054894104910204, 0.020033346447280526]。分别为空间卷积开关、色度对logRGB引导。
- B_appearance_on_minus_off：影-0.021790点，区间[-0.061018002146235165, 0.013726230586496146]。分别为空间卷积开关、色度对logRGB引导。
- B细节特征修正相对RMS均值0.308702483620042；预测复现与权重不变检查通过。
- A/B移除新层后云影IoU均更高，未证明新材质层带来云影收益。保留B关层候选：地表/云/影89.423287/85.091846/65.726375；候选移除material参数后535图逐像素复现。下一步补仅解码器同设置训练。
- 决策：保护地表/云条件下影IoU超过450：AlogRGB引导,Blog色度引导；70%门槛见主表。
- 535图独立评测、107ROI配对区间、同权重干预、3个新参数张量及原解码器更新、其余原参数冻结、初始化/记录批次/权重SHA通过。旧branch_updated字段按组名生成，验收改用独立张量对比，原记录保留。单seed探索。
- 结果：`../outputs/phase476/phase476_20261107/RESULT.md`；网络：`tools/shadow_material476.py`；计划：`research_plans/phase476.md`。

## Phase477：仅解码器续训归因对照（2026-11-07，完成）

- 从450同一起点，只开放原解码器，新增参数0；骨干、450分支、BN冻结。源训练8490/验证535、512²、FP32、RTX4090D、目标监督0%。6000步、batch4、seed45201，lr1e-5、warmup200；与476相同不加权CE及保持约束，固定argmax。
- 目标：核查476关层候选的收益能否仅靠解码器续训获得。共同初始权重与记录批次均与476A/B核验一致；1000/3000/6000及最佳点均独立复算535图。
- 1000步：地表/云/影89.424737/85.142900/65.628737，mIoU80.065458；角色matched_1000。
- 3000步：地表/云/影89.448524/85.161797/65.671555，mIoU80.093959；角色best_shadow,matched_3000,best_protected。
- 6000步：地表/云/影89.427791/85.151751/65.628509，mIoU80.069350；角色last_6000。
- 477best_minus_476B_off：影-0.054821点，107ROI区间[-0.11744052760960635, 0.0008827647118110126]。
- 477_1000_minus_476A_off：影-0.047842点，107ROI区间[-0.10120008978066544, 0.003360711434780387]。
- 477_3000_minus_476B_off：影-0.054821点，107ROI区间[-0.11744052760960635, 0.0008827647118110126]。
- 训练参数1316516，耗时15.21分钟，峰值分配显存4125166080字节。
- 归因：仅解码器即获得三类提升；同3000步较476B关层影低0.054821点，区间跨0，地表/云/mIoU更高。尚未证明材质层训练过程有稳定额外云影收益。后续结构比较采用匹配的仅解码器续训对照，476关层候选继续保留。
- 决策：477未在保护450地表/云条件下超过476关层候选，保留476关层候选及450原始基线。
- 参数更新/冻结、535图独立混淆矩阵、初始化/记录批次和权重SHA通过；单seed探索。
- 结果：`../outputs/phase477/phase477_20261107/RESULT.md`；训练：`tools/train_shadow477.py`；计划：`research_plans/phase477.md`。

## Phase478：原特征路径可塑性对照（2026-11-08，完成）

- A开放原MsRE/refinement、local、output_residual及解码器；B开放原空间CNN、交叉注意力及解码器。新增参数0，VFM、450分支、BN固定；本轮是训练范围对照，不称结构创新。
- 源训练8490/原始验证535、512²、FP32、RTX4090D、目标监督0%；450共同起点，6000步、batch4、seed45201、lr1e-5、warmup200，与477相同不加权CE及保持约束、固定argmax。
- 目标：较477仅解码器获得额外云影收益；保护450地表/云，云影目标70%。1000/3000/6000及最佳点独立535图复算。
- A/1000步：地表/云/影89.424276/85.137846/65.666869，mIoU80.076330；角色matched_1000。
- A/3000步：地表/云/影89.406210/85.113836/65.686569，mIoU80.068871；角色best_shadow,matched_3000,best_protected。
- A/6000步：地表/云/影89.411919/85.140544/65.649585，mIoU80.067350；角色last_6000。
- A训练参数1736008，并行墙钟34.49分钟，峰值分配显存5912971776字节。
- 478A_1000_minus_477_1000：云影+0.038132点，107ROI配对95%区间[-0.024581670836097656, 0.10989443461971289]。
- 478A_3000_minus_477_3000：云影+0.015014点，107ROI配对95%区间[-0.04706204235976834, 0.07012301679877148]。
- 478A_6000_minus_477_6000：云影+0.021077点，107ROI配对95%区间[-0.05444427588601819, 0.09100166169302543]。
- B/1000步：地表/云/影89.446920/85.193213/65.589938，mIoU80.076690；角色best_shadow,matched_1000,best_protected。
- B/3000步：地表/云/影89.419665/85.189788/65.563414，mIoU80.057622；角色matched_3000。
- B/6000步：地表/云/影89.402068/85.185108/65.555584，mIoU80.047586；角色last_6000。
- B训练参数1430180，并行墙钟34.74分钟，峰值分配显存7201275904字节。
- 478B_1000_minus_477_1000：云影-0.038799点，107ROI配对95%区间[-0.13694914983901824, 0.06092303639089559]。
- 478B_3000_minus_477_3000：云影-0.108141点，107ROI配对95%区间[-0.2456067970643911, 0.01038881108315337]。
- 478B_6000_minus_477_6000：云影-0.072925点，107ROI配对95%区间[-0.21585256278265064, 0.04113952431626431]。
- 结论：A最佳3000步云影65.686569%，较同一步477仅高0.015014点，107ROI区间[-0.047062,0.070123]跨0；地表/云各低0.042314/0.047961点。A云影TP较477减少13像素、FP减少3355像素，未体现新增云影找回能力。B最佳1000步云影65.589938%，同1000步较477低0.038799点；3000和6000步也低于对应477，全部云影差值区间跨0。两组最佳点仍保持450地表/云，但均未超过476关层候选65.726375%。
- 决策：本轮未发现扩大原路径训练范围带来可靠的额外云影收益，未达到70%。保留476关层候选和477匹配训练对照；本轮不替换最佳权重，不把解冻训练包装成新结构。后续应回到云影特征构造的结构假设，避免仅扩大解冻范围。
- 参数更新/冻结、共同初始化/记录批次、独立混淆矩阵、权重及传输SHA通过；单seed验证集探索。
- 结果：`../outputs/phase478/phase478_20261108/RESULT.md`；训练：`tools/train_shadow478.py`；计划：`research_plans/phase478.md`。

## Phase479：多尺度暗区包络分支（2026-11-08，完成）

- 在空间CNN后、交互前新增4992参数分支：logRGB的5/13/29尺度暗区RGB差、平均差、通道离散度，与64维detail融合后写入detail/context。A均值参考，B闭运算参考；零末层初始化，A/B初始可学习参数相同。
- 源训练8490/原始验证535、512²、FP32、RTX4090D、目标监督0%；450起点、6000步、batch4、seed45201、分支seed47901、warmup200，新分支lr1e-3/解码器lr1e-5，其余原参数及450分支/BN冻结；损失与477相同，固定argmax。
- 目标：在450地表/云保护线下提高云影并向70%推进；区分477同设置续训、均值对照与形态学参考贡献。1000/3000/6000及最佳点独立535图，最佳点同权重关分支；107ROI配对区间。
- A/on/1000：地表/云/影89.332975/85.060234/65.533252，mIoU79.975487。
- A/on/3000：地表/云/影89.405332/85.162819/65.589419，mIoU80.052523。
- A/on/6000：地表/云/影89.362427/85.131938/65.503439，mIoU79.999268。
- A/off/3000：地表/云/影89.384314/85.013861/65.678266，mIoU80.025480。
- A并行墙钟32.79分钟，峰值分配显存5826744832字节。
- B/on/1000：地表/云/影89.335872/85.062024/65.549438，mIoU79.982444。
- B/on/3000：地表/云/影89.402535/85.159401/65.586331，mIoU80.049422。
- B/on/6000：地表/云/影89.362150/85.129996/65.492292，mIoU79.994813。
- B/off/3000：地表/云/影89.379870/85.005863/65.679864，mIoU80.021866。
- B并行墙钟32.99分钟，峰值分配显存5826744832字节。

结论：这版暗区包络分支没有带来额外云影收益。A、B最佳点均为3000步，云影IoU为65.589419%和65.586331%，分别比同一步477低0.082136和0.085224点。两组地表/云仍高于450保护线，但未超过476关层候选65.726375%。

B相对A：1000步云影高0.016186点，107ROI配对95%区间[0.002434,0.032169]；3000步低0.003088点，区间[-0.012455,0.007079]；6000步低0.011147点，区间[-0.034786,0.017223]。早期微小差异没有维持到最佳点或训练终点，未证明形态学参考有持续优势。

同权重关闭分支后，A/B云影分别升至65.678266%/65.679864%。B开启−关闭云影为-0.093533点，区间[-0.238829,0.053630]跨0；同时地表/云点估计分别增加0.022665/0.153538点。这不是稳定云影收益的证据。B开启后相对关闭多漏23330个云影像素、少误报14741个像素；相对同一步477多漏12391个云影像素、误报多68个像素，主要差距体现为漏检。

B对detail特征的相对RMS变化均值为27.8961%，全部新参数实际更新，说明分支确实参与了网络计算；这不能证明模型有效利用了形态学描述，也不能把结果归因于单独某一通道。6000步A/B相对477的云影差值为-0.125069/-0.136217点，对应场景区间均在0以下；这仍是同一训练seed下的场景重抽样结果。

决策：不保留479作为性能升级，不继续沿这版暗区包络分支加复杂度；保留476关层候选和477匹配训练对照。云影70%目标未达到。本轮排除了这套具体融合方式的收益，不能推出所有形态学方法无效。

- 新参数/解码器更新、其余原参数/BN冻结，共同初始化/记录批次与权重签名验收通过；单seed验证集探索。
- 形态学闭运算/黑顶帽为已有方法，研究本轮网络融合，不声称算子原创。
- 结果：`../outputs/phase479/phase479_20261108/RESULT.md`；模块：`tools/shadow_envelope479.py`；计划：`research_plans/phase479.md`。

## Phase480：云影亚patch重分配（2026-11-08，完成）

- 第一层VFM金字塔前新增15265参数分支：1/16语义与1/4 CNN细节融合，覆盖率预测软门控细节并投影384维；A普通残差，B约束每4×4子格均值为0，末层无bias零初始化。
- 原431+450起点；CloudSEN训练8490/验证535、512²、FP32、4090D、目标监督0%。6000步、batch4、seed45201、分支seed48001，warmup200；新分支lr1e-3/解码器lr1e-5，其余原参数、450分支与BN冻结。
- A/B相同分割保持损失加0.1覆盖率BCE；477没有新分支及辅助损失。固定argmax，保护450地表/云、目标影70%。1000/3000/6000及最佳点独立535图，最佳点同权重关分支、107ROI配对区间及B均值保持复核。
- A/on/1000：地表/云/影89.422359/85.148038/65.670312，mIoU80.080236。
- A/on/3000：地表/云/影89.468857/85.234947/65.848484，mIoU80.184096。
- A/on/6000：地表/云/影89.476166/85.256199/65.826112，mIoU80.186159。
- A/off/3000：地表/云/影89.436958/85.124238/65.782294，mIoU80.114497。
- A并行墙钟26.54分钟，峰值分配显存4196958208字节。
- B/on/1000：地表/云/影89.449885/85.208122/65.759045，mIoU80.139017。
- B/on/3000：地表/云/影89.491820/85.258232/65.822941，mIoU80.190998。
- B/on/6000：地表/云/影89.472216/85.250994/65.833130，mIoU80.185447。
- B/off/6000：地表/云/影89.419180/85.158750/65.650352，mIoU80.076094。
- B并行墙钟26.67分钟，峰值分配显存4201005056字节。

结论：本轮取得小幅正收益，A普通残差与B近似patch均值保持版本均超过477仅解码器参照和此前476关层候选。A最佳3000步地表/云/影为89.468857/85.234947/65.848484%，mIoU80.184096%；B最佳6000步为89.472216/85.250994/65.833130%，mIoU80.185447%。目标监督0%，均满足450地表/云保护线，云影70%尚未达到。

A较同3000步477云影提高0.176930点，107ROI配对95%区间[0.134532,0.223668]；地表/云也分别提高0.020333/0.073150点。较此前476关层最高候选65.726375%，A提高0.122109点。A相对477少漏17825个云影像素、少误报12137个像素。6000步A/B相对同一步477云影分别提高0.197604/0.204621点，场景区间均在0以上。这是新分支加覆盖率辅助监督的整套方案增量，不能单独归因于均值约束。

新分支自身的作用：A在3000步开启−关闭云影为+0.066190点，区间[-0.016958,0.151500]跨0；B在6000步开启−关闭为+0.182778点，区间[0.129036,0.237555]，地表/云也分别提高0.053037/0.092244点，区间均在0以上。B关闭分支时云影为65.650352%。B开启后少漏8855个云影像素、少误报27037个像素，支持这一训练权重下分支的正贡献。

边界证据：B开启相对关闭，3像素内少漏15073、少误报30898；3—8像素少漏1875、少误报10276；8像素外多漏8093、多误报14137。收益集中在边界附近，内部区域仍有损伤。A开启则多漏66971、少误报116477，体现另一种误差权衡。B边界结果与细粒度重分配的假设相符，但不意味着内部错误已解决。

均值约束是否更优：B−A同1000步云影+0.088733点，区间[0.054074,0.124821]；3000步-0.025543点，区间[-0.062010,0.008354]；6000步+0.007017点，区间[-0.023684,0.037835]。尚未证明均值约束带来持续、稳定的额外收益；B同权重开关优势不能直接当作B优于A的证据。

数值审计：原B没有通过预设最大patch均值残差<1e-5的严格GPU验收，实测4.584808e-5，失败日志已保留。定位为注入1×1卷积的TF32舍入：低秩编码的最大块均值5.513430e-7，数学线性投影期望均值1.984798e-7；只对投影关闭TF32复算，最大均值残差1.974404e-7，535图云影IoU65.832944%，较原B仅低0.000186点。原B的全部535图预测复现一致，原训练和原结果均未覆盖。严格投影仅为零训练诊断，不是严格精度重新训练；本轮性能与权重验收通过，原B严格约束项仍记为未通过。

决策：保留480A/3000作为当前最高云影候选，保留480B/6000用于后续边界机制研究，同时保留旧基线。下一步优先多seed复现与拆分覆盖率辅助监督贡献，暂不继续叠加模块。结果是单seed验证集探索，场景重抽样区间不包含训练seed波动。

- 新参数/解码器更新、其余原参数/BN冻结，共同初始化/记录批次与权重签名验收通过；单seed验证集探索。
- 细节引导上采样参考FADE，本轮测试云影覆盖率与patch均值保持组合。
- 结果：`../outputs/phase480/phase480_20261108/RESULT.md`；模块：`tools/shadow_redistribution480.py`；计划：`research_plans/phase480.md`。

## Phase481：Phase480三seed复现（2026-11-08，完成）

- 固定431+450起点；CloudSEN训练8490/原始验证535、107ROI、目标监督0%。复用45201的Phase480 A/B、Phase477 A（C），新增45202/45203，分支初始化48002/48003；新增六次完整6000步训练。
- 原结构、损失、学习率及默认TF32执行不变；A/B新增15265参数并联合训练解码器，C仅训练解码器。原参数/450/BN冻结，同seed A/B初始化和A/B/C记录批次配对。
- 主口径统一6000步，均值±样本标准差（ddof=1，n=3）：
- C：地表/云/影/mIoU 89.434653±0.014528/85.152061±0.021429/65.632863±0.019119/80.073192±0.006269。
- A：地表/云/影/mIoU 89.469353±0.026223/85.242130±0.037566/65.802295±0.022053/80.171259±0.026340。
- B：地表/云/影/mIoU 89.476598±0.019561/85.247638±0.027138/65.833008±0.009653/80.185748±0.012176。
- A-C：云影增量+0.169431±0.036055个百分点，正收益3/3；逐seed云影增量[0.19760366767668813, 0.18189117684458722, 0.12879889719300763]。
- B-C：云影增量+0.200145±0.009883个百分点，正收益3/3；逐seed云影增量[0.20462090776737796, 0.20699713108649576, 0.18881569402105924]。
- B-A：云影增量+0.030713±0.026941个百分点，正收益3/3；逐seed云影增量[0.007017240090689825, 0.025105954241908535, 0.06001679682805161]。
- 结论：A/B对仅解码器C的正收益在三seed复现，B云影均值更高且波动更小，本轮优先保留B；各seed地表/云保护线均通过，70%未达到。最佳点与同权重关分支详见RESULT/MAIN/MULTISEED_RESULTS。
- A/B对C包含新增结构及0.1覆盖率辅助BCE整体效应；仅覆盖固定基座下续训/分支初始化随机性。B仍为Phase480原精度下近似均值约束，不混入严格FP32诊断。
- 训练代码未变、初始权重、更新范围、冻结状态、批次配对、独立535图复算、选定权重及传输签名均通过。
- 结果：`../outputs/phase481/phase481_20261108/RESULT.md`；计划：`research_plans/phase481.md`。

## Phase482：云影占比对比重分配（2026-11-08，完成）

- A在480B基础上将细节权重p换为p−本patch平均p，再保留块均值约束，辅助覆盖率BCE为0.1；B保留原480B结构，将辅助覆盖率BCE设为0。两组15265分支参数，原解码器同时训练。
- 原431+450共同起点，CloudSEN训练8490/原始验证535、107ROI、目标监督0%；6000步、batch4、seed45201/分支seed48001、warmup200，分支lr1e-3/解码器lr1e-5，原骨干/450/BN冻结，FP32张量与原TF32默认执行不变，固定argmax。
- 以同seed同一步480B为直接对照；升级要求地表/云不降且影提升。完整535图独立评测、107ROI配对区间、最佳点关分支、A只关占比对比、边界/内部漏检误报诊断。
- A/1000：地表/云/影 89.429113/85.149856/65.644348，mIoU80.074439；角色['matched_1000']。
- A/3000：地表/云/影 89.446761/85.185554/65.746060，mIoU80.126125；角色['best_shadow', 'matched_3000', 'best_protected']。
- A/6000：地表/云/影 89.447700/85.199848/65.716301，mIoU80.121283；角色['last_6000']。
- A6000步相对480B影-0.116829点，107ROI区间[-0.14495981286020587, -0.09010077421158584]；训练27.22分钟，峰值分配显存4201267712字节。
- B/1000：地表/云/影 89.492790/85.264493/65.830830，mIoU80.196038；角色['matched_1000']。
- B/3000：地表/云/影 89.529899/85.309124/65.978050，mIoU80.272358；角色['best_shadow', 'matched_3000', 'best_protected']。
- B/6000：地表/云/影 89.531547/85.323205/65.952992，mIoU80.269248；角色['last_6000']。
- B6000步相对480B影+0.119863点，107ROI区间[0.07538104237663426, 0.16221028230022402]；训练27.21分钟，峰值分配显存4201005568字节。

本轮保留482B作为新的单seed候选：**新占比对比结构没有提升，去掉覆盖率辅助损失取得正收益。**

固定6000步，482A云影65.716301%，较同一步480B下降0.116829点，107ROI配对95%区间[-0.144960,-0.090101]；地表、云分别下降0.024516、0.051146点。1000和3000步也低于对应480B。A最佳3000步的对比开关本身贡献0.015420点，但这一同权重贡献不能抵消整套方法相对旧B的退步。不保留A作为升级。

482B只取消0.1覆盖率辅助BCE，保留完整细节门控、块均值约束和分割损失，参数量不变。6000步地表/云/影89.531547/85.323205/65.952992%，mIoU80.269248%；较同一步480B分别提高0.059330/0.072211/0.119863点，三类配对场景区间均在0以上。云影少漏14923、少误报3827像素。该正收益来自训练目标简化，本轮没有新增获胜的结构。

482B最佳3000步为89.529899/85.309124/65.978050%，mIoU80.272358%；云影较此前480A最高候选65.848484%提高0.129566点。这个最佳点相对旧480B最佳点多找回39334个云影像素，同时增加27645个误报，与6000步的双向减少不是同一误差状态。最佳权重关闭整个细节分支后云影下降0.325109点，区间[0.264427,0.388719]，说明去掉辅助损失后分支仍在发挥作用。

下一步优先复现482B的多seed表现。当前支持让门控由分割目标端到端学习；无辅助监督后，不将门控值直接解释为校准的云影覆盖率。尚未达到70%，本轮为固定基座下的单seed验证集探索。

- 共同初始化/记录批次、更新/冻结、独立混淆矩阵、权重与传输签名通过。单seed验证集探索，不代替多seed结论；默认TF32下仍按近似块均值约束解释。
- 结果：`../outputs/phase482/phase482_20261108/RESULT.md`；模块：`tools/shadow_redistribution482.py`；计划：`research_plans/phase482.md`。

## Phase483：无覆盖率辅助损失版三seed复现（2026-11-08，完成）

- 网络与482B一致，新增分支15265参数＋原解码器联合训练，辅助BCE为0。固定431＋450起点；目标监督0%，CloudSEN训练8490/原始验证535、107ROI。复用45201，新增45202/45203，分支初始化48002/48003；两次6000步、batch4、warmup200、分支lr1e-3/解码器lr1e-5，原骨干/450/BN冻结，沿用默认TF32，固定argmax。
- 对照：S为480B/481B有辅助损失，C为477/481C仅解码器，N为482B/483B无辅助损失；同seed初始可学习参数和记录批次匹配。
- 固定6000步均值±样本标准差（n=3、ddof=1）：
- C：地表/云/影/mIoU 89.434653±0.014528/85.152061±0.021429/65.632863±0.019119/80.073192±0.006269。
- S：地表/云/影/mIoU 89.476598±0.019561/85.247638±0.027138/65.833008±0.009653/80.185748±0.012176。
- N：地表/云/影/mIoU 89.534103±0.023197/85.322729±0.029523/65.955832±0.005279/80.270888±0.016043。
- N-S云影增量+0.122825±0.005603点，正收益3/3，三类同时提升3/3。
- N-C云影增量+0.322969±0.014133点，正收益3/3，三类同时提升3/3。
- S-C云影增量+0.200145±0.009883点，正收益3/3，三类同时提升3/3。
- 新增seed45202：并行训练含阶段验证27.23分钟，峰值分配显存4201005568字节。
- 新增seed45203：并行训练含阶段验证27.23分钟，峰值分配显存4201005568字节。

**去掉0.1覆盖率辅助BCE的正收益在三个seed上复现；后续采用无辅助损失版作为结构实验对照。**

固定6000步，无辅助版云影IoU为65.955832±0.005279%，有辅助版为65.833008±0.009653%。配对云影增量平均+0.122825±0.005603个百分点，三个seed分别+0.119863、+0.129287、+0.119324点。地表、云平均增加0.057505、0.075091点，三个seed均为三类同时提升；mIoU平均增加0.085140点。这里没有用各seed最佳检查点替代固定终点。

相对仅解码器，无辅助版云影平均提高0.322969点，三类也均在三个seed上提高。两者使用相同分割损失及保持约束，因而这次结构比较不再混入覆盖率辅助BCE的贡献。每个seed最佳权重关闭细节分支后，云影分别下降0.325109、0.332241、0.332821点；分支在推理中的正贡献同样复现。

网络与482B相同，没有新增结构或参数。保留细节门控与近似块均值约束，由分割目标端到端训练，取消覆盖率辅助监督作为默认设置。三seed只反映固定431＋450基座下的续训和分支初始化波动，均在CloudSEN原始验证535张上评测、目标域监督0%。云影仍在65.96%左右，尚未达到70%。

- 模块字节一致、初始参数/记录批次、更新/冻结、独立535图复算、选定权重及签名验收通过。三seed仅覆盖固定基座的续训与分支初始化波动。
- 结果：`../outputs/phase483/phase483_20261108/RESULT.md`；计划：`research_plans/phase483.md`。

## Phase484：双尺度云影修正（2026-11-08，完成）

- 基于无辅助损失482B，保留16像素细节均值约束，新增跨patch区域支路（1/16网格、5×5深度卷积、1×1卷积、插值、零初始化384通道注入）。A增加64像素区域均值约束，B允许区域整体偏移。新增6832参数，细节与区域合计22097参数。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的482B；完整535图独立评测、107ROI配对区间、整体支路关闭及区域支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.493657/85.264975/65.837223，mIoU80.198618；角色['matched_1000']。
- A/3000：地表/云/影 89.531308/85.310587/65.965614，mIoU80.269170；角色['best_shadow', 'matched_3000', 'best_protected']。
- A/6000：地表/云/影 89.536902/85.332471/65.951037，mIoU80.273470；角色['last_6000']。
- A6000步相对482B云影-0.001955点，107ROI区间[-0.012079828892651533, 0.008443738075273531]；训练27.77分钟。
- B/1000：地表/云/影 89.483064/85.263873/65.807246，mIoU80.184728；角色['matched_1000']。
- B/3000：地表/云/影 89.509348/85.299127/65.962479，mIoU80.256984；角色['matched_3000']。
- B/6000：地表/云/影 89.525710/85.326921/65.963748，mIoU80.272126；角色['best_shadow', 'last_6000', 'best_protected']。
- B6000步相对482B云影+0.010755点，107ROI区间[-0.023519909750107183, 0.04656421052664186]；训练27.77分钟。

结论：本轮未产生值得升级的新结构，保留已三seed复现的无辅助版482B/483。A的云影变化−0.001955点；B仅+0.010755点且地表−0.005837点。两组云影107ROI配对区间均跨零，未满足“地表/云不降、云影提升”。最佳云影点也均低于482B的65.978050%。

机制结果：区域均值约束版A的新增区域支路几乎没有云影净收益（最佳点开启−关闭仅+0.000690点）。不约束均值的B并非没有作用：开启区域支路后，云影误报减少405091像素，但漏检增加268849像素，云影IoU反而−0.010945点；地表/云则分别+0.209856/+0.638755点。这是同权重运行时干预，说明该网络依赖区域支路作大范围类别调整，不代表它相对独立训练基线获得这些增益。

内部区域同样存在交换：B在距离云影边界>8像素处减少152375个误报，同时增加41843个漏检；未实现预期的内部漏检、误报同时改善。不能把关支路后的云影分数当作新最优模型，其地表和云性能已下降。

研究决策：本版不升级、不开展多seed扩跑。后续结构应提供已有细节支路之外的新判别证据，避免继续只对同一门控编码扩大感受野。此前未遮挡参照、对应匹配等方向已有失败记录，重新采用前须明确与历史实现的区别。本轮证据只否定当前双尺度实现的增益，不排除所有区域建模结构。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase484/phase484_20261108/RESULT.md`；模块：`tools/shadow_redistribution484.py`；计划：`research_plans/phase484.md`。

## Phase485：云影过渡剖面证据（2026-11-08，完成）

- 基于无辅助损失482B，新增观测RGB的四方向9点连续剖面，6通道中心差分经沿剖面卷积、位置融合及方向聚合后零初始化注入第一层特征。A正常顺序，B固定乱序，同容量、同初始化；新增13776参数，总模块29041参数。均匀颜色证据为零。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的482B；完整535图独立评测、107ROI配对区间、整体支路关闭及剖面支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.489356/85.262509/65.834590，mIoU80.195485；角色['matched_1000']。
- A/3000：地表/云/影 89.530923/85.308842/65.983604，mIoU80.274456；角色['best_shadow', 'matched_3000', 'best_protected']。
- A/6000：地表/云/影 89.534994/85.325348/65.976520，mIoU80.278954；角色['last_6000']。
- A6000步相对482B云影+0.023528点，107ROI区间[0.005559839546094736, 0.0415550936157796]；训练28.31分钟。
- B/1000：地表/云/影 89.489742/85.262055/65.842737，mIoU80.198178；角色['matched_1000']。
- B/3000：地表/云/影 89.531316/85.305822/65.977225，mIoU80.271454；角色['best_shadow', 'matched_3000', 'best_protected']。
- B/6000：地表/云/影 89.525571/85.322947/65.945485，mIoU80.264668；角色['last_6000']。
- B6000步相对482B云影-0.007507点，107ROI区间[-0.07626156617324895, 0.03378400046972593]；训练28.35分钟。

结论：正常顺序A有微弱整体正收益，尚未验证连续剖面作为独立云影判别机制的优势。固定6000步相对482B，地表/云/影分别+0.003447/+0.002142/+0.023528点；云影107ROI配对95%区间[+0.005560,+0.041555]。固定乱序B云影−0.007507点；A−B云影+0.031035点，但区间[−0.003502,+0.094784]跨零，不能确认连续邻接顺序优于固定乱序。单seed结果不替代训练稳定性复现。

A6000步相对独立训练基线，少误报8968像素，但多漏影2493像素；不是两类错误同时改善。A验证最佳3000步云影65.983604%，比482B最佳65.978050%仅高0.005554点。距70%仍约4.02点，未形成突破。

机制检查：A固定6000步，剖面支路开启−关闭，地表+0.010361、云+0.069465、云影−0.065156点；少误报111892像素，但多漏影83371像素。云影差的场景区间[−0.149663,+0.021011]跨零；最佳3000步也是开启后云影下降0.034119点。支路不是完全未被使用，但其推理作用更接近保云并抑制云影预测，尚不能归因为更好的云影识别。

6000步关闭支路后的云影约66.041676%，但云IoU约85.255882%，低于482B的85.323205%；这不满足保护其他类别的要求，不作为新最优模型。端点干预是同权重零训练诊断，不等价于独立训练消融。

研究决策：保留A作为待复现候选，当前正式基线仍用已三seed复现的482B/483；不将本轮写成已成立的结构贡献。下一步若继续此线，应先定位剖面导致的云→影与影→云交换，查清证据在什么场景有效，再决定改变连接方式或证据表达，避免只凭总分微涨继续堆模块。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase485/phase485_20261108/RESULT.md`；模块：`tools/shadow_redistribution485.py`；计划：`research_plans/phase485.md`。

## Phase486：场景遮光变化共识（2026-11-08，完成）

- 基于482B，新增全图加权对数RGB变化方向及局部投影/正交残差。A每场景估计，B固定等比例方向；冻结VFM内容相似度等作软权重，新增8272参数，总模块23537参数。借鉴DC-ShadowNet色度不变性与WACV2025材质一致边缘思想，具体设计与文献边界见LITERATURE.md。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的482B；完整535图独立评测、107ROI配对区间、整体支路关闭及共识证据支路单独关闭、同权重场景方向替换固定方向、边界/内部错误统计。
- A/1000：地表/云/影 89.491725/85.262241/65.837742，mIoU80.197236；角色['matched_1000']。
- A/3000：地表/云/影 89.547045/85.335508/65.979940，mIoU80.287498；角色['matched_3000']。
- A/6000：地表/云/影 89.561449/85.363508/65.979712，mIoU80.301556；角色['best_shadow', 'last_6000', 'best_protected']。
- A6000步相对482B云影+0.026720点，107ROI区间[0.007101853021723237, 0.049569925783738]；训练28.23分钟。
- B/1000：地表/云/影 89.488160/85.252553/65.839534，mIoU80.193416；角色['matched_1000']。
- B/3000：地表/云/影 89.543273/85.328776/65.992383，mIoU80.288144；角色['best_shadow', 'matched_3000', 'best_protected']。
- B/6000：地表/云/影 89.552455/85.360457/65.947521，mIoU80.286811；角色['last_6000']。
- B6000步相对482B云影-0.005471点，107ROI区间[-0.07027136903987881, 0.0338022615664638]；训练28.22分钟。

结论：A整体小幅正收益，核心场景共识机制尚无明确收益。固定6000步相对482B，地表/云/影分别+0.029902/+0.040302/+0.026720点，mIoU+0.032308点；云影107ROI配对95%区间[+0.007102,+0.049570]。B固定方向的终点云影−0.005471点。A−B云影+0.032191点，但区间[−0.004542,+0.101488]跨零，单seed尚不能确认自适应方向的优势。

A比482B少误报9814像素，同时多漏影2587像素，并非漏检与误报同时改善。A同权重把场景方向换成固定方向，三类变化（自适应−固定）为+0.001415/−0.000316/+0.001866点；云影区间[−0.003572,+0.010354]跨零。535张图的方向与等比例RGB方向夹角中位10.7048°、最大34.1949°，说明方向确有变化，但变化没有转化为清晰的云影收益。该夹角在RGB通道空间，不是太阳或空间方位。

A6000步关闭整条新增证据支路后，云影更高0.065097点，云更低0.176169点；开启支路少误报173165像素，却多漏影123833像素。关闭后的云IoU约85.187339%，低于482B的85.323205%，不能把关支路后的高影分数作为满足保护条件的新最优。此结果与485类似：新增图像证据确实参与预测，但主要呈现保云与漏影之间的交换。

选点说明：按训练内batch4验证预先选定的A最佳点是6000步。独立batch1复测时3000/6000的云影为65.979940/65.979712%，顺序发生极小变化（0.000228点）。保留预先选点，不按重载结果重新挑选；主比较统一6000步不受影响。

研究决策：A作为待复现候选归档，正式基线仍保留已三seed验证的482B/483。不继续以当前场景方向估计为核心堆叠模块；本轮未证明这项机制有独立优势，也未接近70%目标。文献提供了合理假设，实验没有验证其在当前RGB与冻结特征条件下的预期作用。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase486/phase486_20261108/RESULT.md`；模块：`tools/shadow_redistribution486.py`；计划：`research_plans/phase486.md`。

## Phase487：交错patch采样融合（2026-11-08，完成）

- 保留482B，新增第二采样网格语义差融合，额外读取同一冻结431骨干，逆移第一层特征后与原网格作差并低秩注入。A平移8像素，B平移16像素，统一屏蔽16像素边缘；新增12688参数，总模块27953参数。部署多一次骨干前向，不是免费增益或预测投票。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的482B；完整535图独立评测、107ROI配对区间、整体支路关闭及第二视图融合支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.505925/85.295701/65.846541，mIoU80.216056；角色['matched_1000']。
- A/3000：地表/云/影 89.555273/85.346479/66.010647，mIoU80.304133；角色['best_shadow', 'matched_3000', 'best_protected']。
- A/6000：地表/云/影 89.558830/85.370034/65.997577，mIoU80.308814；角色['last_6000']。
- A6000步相对482B云影+0.044585点，107ROI区间[-0.013605299696310878, 0.08236052284015756]；训练33.81分钟。
- B/1000：地表/云/影 89.495655/85.270192/65.831900，mIoU80.199249；角色['matched_1000']。
- B/3000：地表/云/影 89.525946/85.305130/65.981927，mIoU80.271001；角色['best_shadow', 'matched_3000', 'best_protected']。
- B/6000：地表/云/影 89.532291/85.329191/65.966319，mIoU80.275934；角色['last_6000']。
- B6000步相对482B云影+0.013326点，107ROI区间[-0.004881856442418453, 0.03149286831813569]；训练33.81分钟。

本轮结论：半patch交错采样有小幅正收益，但额外推理成本不划算，暂不替换482B正式基线。

统一6000步，A相对482B地表、云、云影分别提高0.02728、0.04683、0.04458个百分点，mIoU提高0.03957点；云影漏检减少8429像素，误报增加2941像素。云影107ROI配对95%区间为[-0.01361, 0.08236]点，尚不能确定跨场景稳定收益。

A的验证预选3000步云影IoU为66.01065%，比482B同为3000步的最佳点高0.03260点。统一6000步A比整patch对照B高0.03126点云影IoU，区间[-0.02778, 0.06798]，半patch改变采样相位的解释尚未得到明确支持。

在A预选3000步权重上，仅打开新增融合支路，三类IoU分别增加0.00594、0.02142、0.02406点；云影区间[-0.01282, 0.06040]。这次支路开启的云影变化方向为正，但幅度仍小。该开关诊断不是独立训练消融。

新增12688参数，但共享骨干仍需额外执行一次：单张512图、batch1的GPU前向中位时间从30.47ms增加到45.26ms，约增加48.5%。当前实现额外跑完整骨干，可优化提前退出，但本轮不改动已训练实现。就当前测量而言，这个成本不足以支撑0.045点云影增量。

本轮验证的是补充patch采样相位是否有效，未直接增加云影物理判别证据。保留结果作为采样瓶颈诊断，不把它当作达到70%目标的主结构突破。所有结果来自原始CloudSEN验证集，目标域监督比例0%，无阈值调整；单seed探索未证明训练稳定性。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase487/phase487_20261108/RESULT.md`；模块：`tools/shadow_redistribution487.py`；计划：`research_plans/phase487.md`。

## Phase488：亮度归一化纹理比较（2026-11-08，完成）

- 保留482B细节支路，新增局部纹理结构张量与环形邻域比较。12/28像素尺度提取纹理方向、能量及邻域差异，卷积编码后零投影注入第一层特征。A梯度按局部亮度加0.02归一化，B原始梯度。新增8272参数，总模块23537参数。不增加骨干前向；显示RGB上计算，不宣称严格光照不变。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的482B；完整535图独立评测、107ROI配对区间、整体支路关闭及纹理支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.500583/85.274494/65.853184，mIoU80.209420；角色['matched_1000']。
- A/3000：地表/云/影 89.566418/85.352240/66.040166，mIoU80.319608；角色['best_shadow', 'matched_3000', 'best_protected']。
- A/6000：地表/云/影 89.541658/85.349434/65.996755，mIoU80.295949；角色['last_6000']。
- A6000步相对482B云影+0.043763点，107ROI区间[-0.021756628506318165, 0.0893707769330941]；训练28.09分钟。
- B/1000：地表/云/影 89.496270/85.271350/65.850366，mIoU80.205995；角色['matched_1000']。
- B/3000：地表/云/影 89.541371/85.323016/66.017933，mIoU80.294107；角色['best_shadow', 'matched_3000', 'best_protected']。
- B/6000：地表/云/影 89.548588/85.345827/66.014230，mIoU80.302882；角色['last_6000']。
- B6000步相对482B云影+0.061238点，107ROI区间[0.04408973132434344, 0.0816992785181739]；训练28.09分钟。

本轮结论：局部纹理与环形邻域比较得到小幅正收益；亮度归一化不是已证实的收益来源。保留原始梯度B为待复现候选，暂不替换经过多seed验证的482B正式基线。

统一6000步，B相对482B地表/云/云影提高0.01704/0.02262/0.06124点，mIoU提高0.03363点。云影漏检减少7156像素、误报减少2659像素；107ROI配对95%区间[0.04409,0.08170]点。A云影提高0.04376点，区间[-0.02176,0.08937]，比B低0.01747点。故不能把亮度归一化作为确定有效的机制。

训练验证预选的最佳点均为3000步，A云影66.04017%、B66.01793%，482B为65.97805%。A在该点的云影增量0.06212点，但到6000步回落；主结论继续以统一6000步为准，不因最佳点排序改变而改口径。

在各自预选3000步的同一权重上，仅打开新纹理支路，A地表/云/云影提高0.03434/0.06512/0.07318点；B提高0.03130/0.05575/0.11038点。B云影增量区间[0.08015,0.14074]，漏检减少10533像素、误报减少8425像素。这比最近几轮支路关闭后云影反而变好的情况更有说服力：纹理支路确实参与了正确判别。但开关诊断不是独立训练消融，不能把0.11038点当作相对482B的增量，也不能据此认定修复的是水体或薄影。

新增8272参数，不增加骨干前向。A单张512图、batch1的GPU前向中位时间33.93ms，482B31.53ms，增加7.61%；本次计时针对A，不冒充B实测。B同网络规模、少一次亮度归一化，但其速度仍需单独测量。

原始CloudSEN验证535张、107ROI，目标监督0%，无阈值调整。场景区间不涵盖训练seed波动，也未校正连续多轮架构筛选；当前是小增量候选，距云影70%目标仍约4个百分点。下一步优先复现B的同6000步收益，再决定是否基于纹理证据扩展结构。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase488/phase488_20261108/RESULT.md`；模块：`tools/shadow_redistribution488.py`；计划：`research_plans/phase488.md`。

## Phase489：Phase488 B三seed复现（2026-11-08，完成）

- 原始梯度纹理支路与488逐字节一致；固定431＋450起点，复用45201/48001，新增45202/48002、45203/48003，与482B/483同seed配对。每组6000步、batch4、分支lr1e-3/解码器1e-5；骨干/450/BN固定；CloudSEN训练8490/原始验证535、目标监督0%，固定argmax。
- 固定6000步均值±样本标准差：
- S：地表/云/影/mIoU 89.534103±0.023197/85.322729±0.029523/65.955832±0.005279/80.270888±0.016043。
- N：地表/云/影/mIoU 89.548464±0.020018/85.337662±0.028462/66.019581±0.010858/80.301903±0.012618。
- N−S云影增量+0.063749±0.005589点，正收益3/3，三类同时提升3/3。

结论：Phase488 B的小幅正收益在三个seed上复现，三类IoU均为3/3提升，可作为当前CloudSEN域内结构研究的后续基线。

统一6000步，云影IoU由65.95583±0.00528提高到66.01958±0.01086%；逐seed配对增量为+0.06124、+0.05986、+0.07015个百分点，平均+0.06375±0.00559点。地表、云、mIoU配对平均增量分别为+0.01436、+0.01493、+0.03101点，三次均为正。

纹理支路的同权重开关诊断也在三个seed上保持正方向：预选检查点云影分别提高+0.11038、+0.17735、+0.13948点，地表和云也全部提高。其中45201预选3000步，另两个seed预选6000步；这属于机制辅助证据，不混入统一6000步的主要收益，也不等同于独立训练消融。

这轮支持“原始梯度纹理及环形邻域比较提供可复现的小增量”，不支持“已经解决云影识别瓶颈”。云影均值仍距70%约3.98个百分点。后续可以保留该分支继续研究，不能把0.064点写成明显跃升。多seed覆盖的是固定431＋450起点下的分支初始化与续训随机性，未验证跨域泛化。

验收恢复记录：首次归档检查失败来自phase编号替换误改了预期源权重SHA中的483片段。单独的accept_recheck489.py只修复该哈希常量；原始锁定文件、训练实现、权重及评测均未修改。真实源SHA、全部初始代码签名和训练冻结检查随后通过，见ACCEPTANCE_RECOVERY.json。原失败记录保留供追踪。

- 代码、公共初始化/记录批次、更新冻结、独立535图复算、权重签名验收通过。三seed仅覆盖固定基座续训随机性。
- 结果：`../outputs/phase489/phase489_20261108/RESULT.md`；计划：`research_plans/phase489.md`。

## Phase490：纹理相容方向邻域比较（2026-11-08，完成）

- 保留488B原始梯度纹理分支，新增16/32像素偏移的8方向邻域比较。读取相对亮度、纹理能量差、纹理方向差、语义差；A按冻结语义及纹理相容性加权，B均匀加权，卷积编码后零投影注入第一层特征。新增7696参数，总模块31233参数，不增加骨干前向。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的488B；完整535图独立评测、107ROI配对区间、整体支路关闭及新方向邻域支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.495457/85.264853/65.862967，mIoU80.207759；角色['matched_1000']。
- A/3000：地表/云/影 89.530540/85.316456/66.013863，mIoU80.286953；角色['matched_3000']。
- A/6000：地表/云/影 89.531154/85.333108/66.019498，mIoU80.294586；角色['best_shadow', 'last_6000', 'best_protected']。
- A6000步相对488B云影+0.005268点，107ROI区间[-0.017545928344063454, 0.027433383938333465]；训练32.00分钟。
- B/1000：地表/云/影 89.496500/85.269321/65.856198，mIoU80.207340；角色['matched_1000']。
- B/3000：地表/云/影 89.529225/85.313972/66.017791，mIoU80.286996；角色['matched_3000']。
- B/6000：地表/云/影 89.528974/85.327019/66.019437，mIoU80.291810；角色['best_shadow', 'last_6000', 'best_protected']。
- B6000步相对488B云影+0.005207点，107ROI区间[-0.021721624881341838, 0.03285270751942554]；训练32.00分钟。

结论：本轮方向邻域扩展未形成有效升级，保留已通过三seed复现的488B基线，不合并490结构。

统一6000步，A相对488B地表/云/云影增量为-0.017435/-0.012719/+0.005268点；B为-0.019615/-0.018808/+0.005207点。两者均损伤地表和云，云影增量接近零且107ROI配对95%区间跨零。A相对B云影仅+0.000061点，区间[-0.016160,0.017154]，相容性加权没有优于均匀读取的明确证据。

同权重单独开启新增方向邻域支路，A云影下降0.077870点，B下降0.082146点；A误报减少213098像素，但漏检增加152179像素。新增支路更倾向压制云影误报，同时牺牲召回。A云IoU同时提高0.201762点，故简单关闭新支路并不是不损伤其他类的升级方案。开关诊断涉及联合训练后的权重，不能直接推断独立训练消融的结果。

新增7696参数，但邻域证据计算仍有成本：同一512图batch1 GPU前向中位时间从488B的34.01ms增加至490A的41.65ms，增加22.46%。当前实现的计算开销与几乎为零的云影增量不相称。

本轮不支持“在488B上继续增加冻结语义/纹理相容性邻域选择就能明显提升云影”的假设。它不否定488B已经复现的局部纹理收益。后续不继续堆叠同类邻域分支；保留488B，寻找能够区分漏检云影与真实暗地表的新证据或不同交互位置。原始CloudSEN验证535张、目标域监督0%、固定argmax；本轮单seed探索，不作跨域结论。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase490/phase490_20261108/RESULT.md`；模块：`tools/shadow_redistribution490.py`；计划：`research_plans/phase490.md`。

## Phase491：浅深层关系交互（2026-11-08，完成）

- 保留488B，读取第1及第4交互位置输出的同网格特征，分别LayerNorm并投影到16维。A拼接浅深差分及逐通道乘积，B直接拼接两层投影作同参数对照，经卷积与零投影注入第一层特征。新增19088参数，总模块42625参数，不增加骨干前向。没有将乘积直接解释成真实材质保持量。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的488B；完整535图独立评测、107ROI配对区间、整体支路关闭及新跨层支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.498032/85.273564/65.849264，mIoU80.206953；角色['matched_1000']。
- A/3000：地表/云/影 89.559853/85.348454/66.024314，mIoU80.310874；角色['best_shadow', 'matched_3000', 'best_protected']。
- A/6000：地表/云/影 89.543429/85.343564/66.021493，mIoU80.302829；角色['last_6000']。
- A6000步相对488B云影+0.007263点，107ROI区间[-0.013860966615354542, 0.024542240850076506]；训练28.31分钟。
- B/1000：地表/云/影 89.495125/85.268671/65.857070，mIoU80.206956；角色['matched_1000']。
- B/3000：地表/云/影 89.540592/85.320117/66.016593，mIoU80.292434；角色['matched_3000']。
- B/6000：地表/云/影 89.539836/85.332463/66.032024，mIoU80.301441；角色['best_shadow', 'last_6000', 'best_protected']。
- B6000步相对488B云影+0.017794点，107ROI区间[-0.0015363410847669455, 0.037895546199235484]；训练28.31分钟。

结论：本轮浅深层关系交互未超过488B，保留488B基线，不合并491结构。

统一6000步，A显式差分乘积相对488B地表/云/云影增量为-0.005160/-0.002263/+0.007263点；B直接拼接为-0.008752/-0.013364/+0.017794点。两者均轻微损伤地表和云，mIoU没有提升。A云影107ROI区间[-0.013861,0.024542]，B[-0.001536,0.037896]，均跨零。A相对B云影低0.010531点，不能支持显式关系交互优于直接多层输入。

训练验证预选点A为3000步、B为6000步。只打开新增跨层支路时，A的地表/云/云影分别变化-0.004202/-0.000555/-0.006112点；B为-0.022399/+0.069936/-0.046520点。两者都未显示新增支路对云影的明确直接正贡献。A关闭支路的诊断分数略好，但这是联合训练后关闭模块，不等同于单独训练的新结构，也不足以替代已通过三seed复现的488B。

新增19088参数，不增加骨干前向；同一512图batch1 GPU前向中位时间34.52ms，488B为34.10ms，增加约1.21%。成本较低，但这不能弥补收益不足。

本轮检验的是“浅深层低秩投影后的差分与乘积”这一具体实现，未证实它能读取地物语义保留的云影证据，也不据此否定所有跨层建模方法。当前结论限于原始CloudSEN验证535张、目标监督0%、单seed6000步，固定argmax。

启动记录：首个尝试在训练前因phase编号替换将训练集数量断言8490误写为8491而退出，未产生任何训练步或权重。已修复为数字边界限定的编号替换，并完整重启两组；本报告仅使用修复后完成的结果。首次失败代码与记录保存在startup_failure_001，正式训练代码签名始终保持不变。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase491/phase491_20261108/RESULT.md`；模块：`tools/shadow_redistribution491.py`；计划：`research_plans/phase491.md`。

## Phase492：颜色方向遮光边缘证据（2026-11-09，完成）

- 保留488B，新增颜色梯度分解支路：在12/28像素尺度上，把RGB空间梯度分成沿颜色方向的平行分量、垂直残差及能量占比。A采用局部RGB方向，B固定灰色方向，编码后零投影注入第一层。新增8272参数，总模块31809参数，不增加骨干前向；显示RGB仅作学习线索，不作严格反射率解释。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的488B；完整535图独立评测、107ROI配对区间、整体支路关闭及新颜色梯度支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.501004/85.268687/65.892008，mIoU80.220566；角色['matched_1000']。
- A/3000：地表/云/影 89.531540/85.310649/66.029799，mIoU80.290663；角色['matched_3000']。
- A/6000：地表/云/影 89.539430/85.333583/66.045598，mIoU80.306204；角色['best_shadow', 'last_6000', 'best_protected']。
- A6000步相对488B云影+0.031368点，107ROI区间[0.008547950780955205, 0.05363383948589371]；训练28.49分钟。
- B/1000：地表/云/影 89.501082/85.269704/65.875506，mIoU80.215431；角色['matched_1000']。
- B/3000：地表/云/影 89.534817/85.313089/66.037408，mIoU80.295104；角色['matched_3000']。
- B/6000：地表/云/影 89.540142/85.336555/66.048451，mIoU80.308383；角色['best_shadow', 'last_6000', 'best_protected']。
- B6000步相对488B云影+0.034221点，107ROI区间[0.013124791448506912, 0.05577839631619063]；训练28.48分钟。

结论：颜色梯度分解带来小幅云影正收益，但没有保住地表和云；局部颜色方向未优于固定灰色方向。本轮不替换488B基线。

统一6000步，A相对488B地表/云/云影增量为-0.009159/-0.012244/+0.031368点，mIoU约+0.003322点；B为-0.008446/-0.009272/+0.034221点，mIoU约+0.005501点。A云影107ROI配对95%区间[0.008548,0.053634]，B[0.013125,0.055778]。云影增量方向明确，但幅度小且三类同时提升条件未满足。

A云影漏检减少292像素、误报减少6468像素；B漏检减少1981像素、误报减少4539像素。A比B云影低0.002853点，区间[-0.014747,0.008542]，不能归因于局部颜色方向符合遮光假设；固定灰色方向同样有效，且点估计略好。

两组预选最佳均为6000步。同权重单独开启新增颜色梯度分支时，A地表/云/云影增加0.073435/0.247153/0.029541点；B增加0.070320/0.243442/0.033422点。云影开关区间较宽且跨零；支路能影响预测，但不据此宣布云影机制已证实。相比关闭，A漏检增加141534像素、误报减少220901像素，表明联合训练后该支路的主要动作仍是压制误报并牺牲部分召回。

新增8272参数，不增加骨干前向。A单张512图batch1 GPU前向中位时间35.54ms，对照488B为34.35ms，增加3.45%；本次计时仅针对A。单seed、原始CloudSEN验证535张、目标监督0%、固定argmax，未作跨域结论。

保留本轮作为“颜色梯度可能有弱收益、局部颜色轴假设尚未成立”的证据。后续不继续仅调整颜色方向或叠加同类末端分支；保持已通过多seed复现的488B作为结构研究基线。

归档：完整权重保留在远端 /home/scv/shared/phase492_20261109，已完成实际权重验签；本地完整归档评测结果及签名，未完成权重下载，.partial 文件不可用于推理。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase492/phase492_20261109/RESULT.md`；模块：`tools/shadow_redistribution492.py`；计划：`research_plans/phase492.md`。

## Phase493：下采样前亮暗次序纹理（2026-11-09，完成）

- 保留488B，新增下采样前亮暗次序纹理支路：原图灰度在1/2/4像素半径比较八邻域，先编码再4倍池化，并与周围12–28像素环形邻域比较。A取亮度差符号，B取tanh(20×亮度差)，同参数对照。48通道经卷积和零投影注入第一层，新增13456参数，总模块36993参数，不增加骨干前向。次序编码参考[Census局部变换](https://woodfill.com/Papers/Census94.pdf)，本轮检验其在云影网络中的作用。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的488B；完整535图独立评测、107ROI配对区间、整体支路关闭及新次序纹理支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.515966/85.291481/65.949563，mIoU80.252336；角色['matched_1000']。
- A/3000：地表/云/影 89.550257/85.336977/66.080985，mIoU80.322740；角色['best_shadow', 'matched_3000', 'best_protected']。
- A/6000：地表/云/影 89.557115/85.362379/66.079468，mIoU80.332987；角色['last_6000']。
- A6000步相对488B云影+0.065237点，107ROI区间[0.008566206009626143, 0.10839120525883102]；训练29.15分钟。
- B/1000：地表/云/影 89.517830/85.287759/65.934526，mIoU80.246705；角色['matched_1000']。
- B/3000：地表/云/影 89.555257/85.333881/66.070040，mIoU80.319726；角色['matched_3000']。
- B/6000：地表/云/影 89.560528/85.355255/66.081382，mIoU80.332389；角色['best_shadow', 'last_6000', 'best_protected']。
- B6000步相对488B云影+0.067152点，107ROI区间[0.040803712362262344, 0.0929379570336703]；训练29.15分钟。

结论：本轮得到三类同时小幅提升的正结果。统一6000步，A/B相对488B云影分别增加0.065237/0.067152个百分点，地表与云的点估计也均提高。保留为候选结构，尚不替换已完成多seed复现的488B。

A三类增量为+0.008527/+0.016552/+0.065237点，mIoU增加0.030106点；B为+0.011940/+0.009428/+0.067152点，mIoU增加0.029507点。云影107ROI配对95%区间：A [0.008566,0.108391]，B [0.040804,0.092938]。地表与云区间仍跨零，因此“三类同时提升”描述的是本轮点估计，不代表三类增益均已得到统计确认。

相对同6000步488B，A云影漏检减少6337像素、误报减少4774像素；B漏检减少7051像素、误报减少4115像素。这轮对云影的总收益来自两类错误同时减少，并非仅扩大或压缩云影区域。

A的亮暗次序没有胜过B的亮度差幅度：A−B云影为-0.001915点，区间[-0.051047,0.037998]；A的mIoU仅比B高0.000599点。结果支持继续研究这套细粒度邻域比较分支，但不能归功于次序编码的光照不变性。当前对照也未单独隔离“先比较再下采样”与增加容量的作用；不能把它宣布为已确认的唯一机制。

新增分支开关对照：A在训练验证预选的3000步开启−关闭，地表/云/云影为+0.114207/+0.298692/+0.205155点，云影区间[0.055385,0.341928]；B在6000步为+0.079459/+0.196174/+0.224176点，云影区间[0.127158,0.310467]。两组关闭新增分支均明显变差，说明分支在各自训练后的网络中有实际作用。A/B开关实验检查点不同，不能用0.205与0.224直接判定哪种编码更优，也不能将开关增量当作相对488B的收益。

同权重开关下，A云影漏检增加76630、误报减少161720；B漏检增加37720、误报减少106787。分支在联合训练后主要压制误报，同时牺牲部分召回；这与整套重训网络相对488B同时减少漏检和误报，是两种不同对照。

新增13456参数，不增加骨干前向。A实测单张512图batch1 GPU前向中位耗时38.76ms，对照488B为33.96ms，增加14.15%。本次只测A，不能代替B的耗时。两组并行训练均约29.15分钟，不含后续独立评测与消融。

研究决定：这是值得复现的小增益，不是接近70%目标的突破。优先将493B列为下一轮多seed候选，保留A作为编码对照；本轮A/B差异不足以证明B更优。所有结果来自原始CloudSEN验证535张、单seed、目标监督0%、固定argmax，不作跨域提升结论。

归档：完整权重保留在远端 /home/scv/shared/phase493_20261109，已完成实际权重验签；本地保存完整评测元数据、报告及签名，不下载权重。训练代码和训练协议全程保持锁定。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase493/phase493_20261109/RESULT.md`；模块：`tools/shadow_redistribution493.py`；计划：`research_plans/phase493.md`。

## Phase494：493B三seed复现（2026-11-09，完成）

- 不改结构、损失或决策；复用493B的45201/48001，新增45202/48002、45203/48003，与488B/489对应seed逐一配对。
- 固定431＋450起点，6000步、batch4；CloudSEN训练8490/原始验证535，目标监督0%；原骨干/450/BN冻结，固定argmax。
结论：493B的小幅正收益完成三seed复现。固定6000步，地表、云、云影及mIoU均为3/3正收益，三类同时提升也是3/3。将493B/494作为后续结构研究的精度对照基线。

三seed均值±样本标准差：488B/489地表89.548464±0.020018、云85.337662±0.028462、云影66.019581±0.010858、mIoU80.301903±0.012618；493B/494分别为89.564801±0.022684、85.353423±0.028073、66.071501±0.012895、80.329908±0.013756。

逐seed云影增量：45201为+0.067152点，新增45202为+0.044477点，新增45203为+0.044130点。配对平均增量为+0.051920±0.013193点；地表、云、mIoU配对平均增量分别为+0.016337、+0.015761、+0.028006点。两个新增seed均复现了正方向，但幅度小于首次探索的0.067点。

云影漏检/误报在三个seed上均同时减少：45201减少7051/4115像素；45202减少3611/4337像素；45203减少5103/1962像素。配对图像索引、场景编号和基准预测混淆矩阵一致，三组结果使用同一535张验证图及标签口径。

三个seed预选最佳云影点均为6000步。开启新增分支相对关闭，云影分别增加0.224176、0.229024、0.270688点，地表和云也均增加。说明新增分支在训练后的网络中有实际作用；这些开关增量不能当作相对488B的收益，也不能代替独立训练的机制消融。

本轮未改网络：保留488B，并采用493B的亮度差幅度编码，在原图1/2/4像素偏移比较，tanh(20×差值)后再池化，再与环形邻域比较，卷积编码后注入首层特征。新增13456参数。当前证据支持这套完整结构的小增益；不证明亮暗次序不变性，也未单独证明“先比较再下采样”是唯一有效因素。

B版本单张512图、batch1、交替各30次GPU前向计时：488B中位34.73ms，493B中位39.86ms，增加14.77%。这是GPU前向计时，不包含IO或预处理；新增两组并行训练各约29.20分钟，不含独立评测与消融。

后续采用493B/494作为精度研究对照，保留本轮全部逐seed结果。三seed云影均值66.0715%，距离70%仍有3.9285个百分点；本轮属于可重复的小增益，尚未实现70%目标。

范围：431＋450固定起点，CloudSEN训练8490、原始验证535、107ROI，目标监督0%，固定argmax；只检验续训采样与分支初始化随机性，不包含基座重新训练或新数据划分。完整权重已在远端 /home/scv/shared/phase494_20261109 验签保存；45201复用 /home/scv/shared/phase493_20261109/B/network_run。本地保存评测、报告及签名。

- 逐seed固定6000步增量（地表/云/影/mIoU）：
  - 45201: +0.011940/+0.009428/+0.067152/+0.029507
  - 45202: +0.020897/+0.019386/+0.044477/+0.028253
  - 45203: +0.016173/+0.018469/+0.044130/+0.026257
- 模块逐字节相同、历史引用签名、公共初始化、记录批次、更新冻结、独立评测混淆矩阵及远端权重验收通过。
- 结果：`../outputs/phase494/phase494_20261109/RESULT.md`；计划：`research_plans/phase494.md`。

## Phase495：细纹理写回MsRE主干token（2026-11-09，完成）

- 保留493B全部输出路径，共享已有16通道细纹理编码，池化到token网格并通过零初始化16→384投影写入MsRE输出。A在第1次交互写入并经过后续9个冻结VFM块，B在第4次交互写入；均按当前token的脱离梯度RMS匹配量纲。新增6144参数，总结构模块43137参数，无额外骨干前向，但A需要反向穿过后续冻结块。
- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。
- 直接对照为同seed、同6000步的493B；完整535图独立评测、107ROI配对区间、整体支路关闭及新token写回支路单独关闭、边界/内部错误统计。
- A/1000：地表/云/影 89.513046/85.281309/65.930081，mIoU80.241479；角色['matched_1000']。
- A/3000：地表/云/影 89.540836/85.308251/66.031771，mIoU80.293619；角色['matched_3000']。
- A/6000：地表/云/影 89.543811/85.323812/66.072661，mIoU80.313428；角色['best_shadow', 'last_6000', 'best_protected']。
- A6000步相对493B云影-0.008721点，107ROI区间[-0.08848319439254197, 0.0738054651619249]；训练33.07分钟。
- B/1000：地表/云/影 89.513246/85.276328/65.930987，mIoU80.240187；角色['matched_1000']。
- B/3000：地表/云/影 89.533205/85.319834/66.047933，mIoU80.300324；角色['matched_3000']。
- B/6000：地表/云/影 89.536382/85.321472/66.053944，mIoU80.303933；角色['best_shadow', 'last_6000', 'best_protected']。
- B6000步相对493B云影-0.027439点，107ROI区间[-0.08356461812967755, 0.03932232513420776]；训练29.86分钟。

结论：本轮细纹理写回主干token没有带来正收益，A/B三类IoU点估计均低于493B。保留已通过三seed复现的493B/494作为结构研究基线，不采用本轮写回版本。

主比较固定seed45201、6000步。493B地表/云/云影为89.560528/85.355255/66.081382，mIoU80.332389。A第一交互后写入为89.543811/85.323812/66.072661，mIoU80.313428；B第四交互后写入为89.536382/85.321472/66.053944，mIoU80.303933。这里使用493B同seed结果，不与其三seed均值混比。

A相对493B三类变化为-0.016717/-0.031443/-0.008721点，mIoU变化-0.018961点；B为-0.024146/-0.033783/-0.027439点，mIoU变化-0.028456点。云影107ROI配对95%区间分别为[-0.088483,0.073805]和[-0.083565,0.039322]，均跨零。结果未提供增益证据，也不据此宣布这类写回结构普遍有害。

A比B云影高0.018718点，但区间[-0.069553,0.102004]跨零；本轮不足以认定早期写入优于晚期写入。接入位置同时改变后续VFM传播、局部反馈及影响的输出尺度，本轮没有将这些因素逐一分离。

两组训练验证预选最佳点均为6000步。只开启新增写回路径相对关闭，A地表/云/云影变化为-0.016014/+0.030079/-0.072515点，B为+0.219005/+0.471019/-0.098898点。即在各自训练后的权重上，关闭写回路径反而使云影点估计回升；但A关闭后云IoU下降，B关闭后地表和云均下降，不能将关闭版本当成满足三类保护要求的新解。

写回开关的云影区间也跨零：A [-0.208216,0.061204]，B [-0.462385,0.266318]。A开启后云影漏检减少6206像素、误报增加25323像素；B漏检减少92362、误报增加161355，体现了本轮写回路径的召回/误报交换。相对独立训练的493B则是另一种对照：A云影漏检增加4479、误报减少4859；B漏检增加8021、误报减少6101，均没有提升云影IoU。

新路径并未失效或停在零初始化：真实参数更新与冻结验收通过，每次前向恰好写入一次；6000步最后一个记录训练批次中，写回更新相对token的RMS比例约为A 1.8721%、B 3.6483%。这些比例仅证明路径参与，不能当作机制有效性的指标。原VFM、431原有适配器、450分支及BN参数保持冻结，493B保留模块与新增写回投影共同训练，A反向梯度穿过冻结块到达结构模块。

新增6144参数，总结构模块43137参数。两组并行训练含阶段验证分别为A 33.07分钟、B 29.86分钟；峰值分配显存约A 5.80GB、B 4.29GB。cost_both495.py在同一张512图、batch1下各预热5次、交替各30次GPU前向：493B中位38.76ms，A 39.32ms（+1.45%），B 39.36ms（+1.57%）。不含IO或预处理，不代表完整训练吞吐。

研究决定：不继续仅调整这条全图写回路径的接入层号。已被验证有效的输出端细纹理证据，进入主干后仍需重新验证，不能假定传播更早就一定更有效。本轮保留为否定当前实现的实验记录，不把全部细节模块关闭的损伤归给新增写回路径。

训练内best_protected沿用450门槛记录；是否升级基线只看同6000步相对493B的三类变化。所有结果来自原始CloudSEN验证535张、107ROI、单seed、目标监督0%、固定argmax。完整权重已在远端 /home/scv/shared/phase495_20261109 验签保存，本地保存报告、评测元数据和签名。

- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。
- 结果：`../outputs/phase495/phase495_20261109/RESULT.md`；模块：`tools/shadow_redistribution495.py`；计划：`research_plans/phase495.md`。

## 后续维护规则

从 Phase 16 开始，每个 Phase 完成后在本文件末尾追加以下内容：

1. 简介与研究动机。
2. 明确、可证伪的实验目标和通过门槛。
3. 相对上一候选的网络、训练或部署改动。
4. 数据集、输入尺寸、精度、batch、硬件、warmup/iterations 等关键设置。
5. 完整分割指标或数值一致性指标，以及参数量、产物大小、延迟、吞吐和显存中实际测得的部分。
6. 与直接基线的差值、是否通过、失败原因和下一步决策。

禁止用局部样本 parity 代替完整数据集 mIoU，禁止把 fake quant 结果表述为真实 INT8 加速，禁止混用不同精度或不同计时口径而不加说明。

后续所有 Phase 及其分支统一追加到本文件的维护规则之前，不再新增 `LOG*.md`。历史文件名仅用于追溯归档来源；Phase 63 等研究计划中记录路径以本文件对应阶段为准。


## Phase496 — 移除MsRE后完整40K重训（2026-11-10，已完成）

- 原431融合结构对照 vs 无MsRE/LocalReadWrite的新纹理适配器（493B幅度微纹理、488B环形纹理、480/482子patch重分配）。保留空间CNN、交互、输出残差和分割头；无450分支。
- 同一官方DINOv3-S+/16冻结基座；任务模块全新初始化。CloudSEN8490训练/535验证，目标监督0%，seed42901、batch2、AdamW 1e-4、40K并行重训。两组训练及四个检查点重载复评完成。
- 固定40K：原结构地表/云/云影/mIoU=88.9381/84.5160/64.2838/79.2460；新结构=89.2964/85.1042/64.9683/79.7896。差值=+0.3582/+0.5882/+0.6845/+0.5436点，三类均正；云影尚未达到70%。
- 原结构验证最优mIoU79.3505；新结构最优即40K的79.7896。单seed完整替代包结果，尚无多seed稳定性结论。
- 可训练参数1,850,952→1,494,597（−19.25%）；GPU前向25.269→30.188ms（+19.47%，batch1/512/FP32，不含IO与预处理）。
- 新组关断整个纹理模块后云影58.5879、mIoU76.4605；开启分别+6.3804/+3.3291点。这是推理关断，不是移除后重训消融。
- 10步预检、初始共同层/VFM签名、全部21个纹理参数更新、全部抽查批次至40K、冻结权重与代码锁均通过。两组并行训练加主复评墙钟8880.71秒。
- 结果：`outputs/phase496/phase496_20261110/RESULT.md`；远端权重与代码：`/home/scv/shared/phase496_20261110`。本地元数据哈希验收通过。


## Phase497 — 同材质候选参照（2026-11-10，已完成）

- 在496无MsRE网络中，用16维语义相似度+可学习空参照替换固定环形参照；16/32像素各8方向，比较亮度、色度、纹理方向/能量和微纹理幅度。保留子patch细节重分配。
- 单seed42901，CloudSEN8490训练/535验证，目标监督0%，官方DINOv3-S+/16冻结，任务模块全新初始化。用户约定：后续结构探索不做多seed复现。
- 远端环境运行中重启，日志止于28,600步；从28K完整检查点恢复至40K。模型、优化器、调度器恢复逐项一致；30K至40K抽查批次均匹配496。原检查点未保存RNG状态，不宣称与不中断运行逐位等价。原训练代码未更改，恢复程序另行锁定签名。
- 固定40K：496地表/云/影/mIoU=89.2964/85.1042/64.9683/79.7896；497=89.1201/85.0683/64.9218/79.7034。差值−0.1762/−0.0359/−0.0465/−0.0862点。
- 云影FN增加86,728，FP减少122,943，未提高IoU。497验证最优mIoU79.7106（30K），云影64.6013。
- 推理均匀候选（保留学习到的空参照置信度）云影64.9282，完整学习参照64.9218，差−0.0064点；学习选择未体现额外云影收益。关闭整个纹理模块降到58.4256，但不能把纹理模块整体作用归因于新增选择机制。
- 可训练参数+7,298（+0.49%）；本轮同条件计时496=28.039ms、497=28.485ms，+1.59%。
- 结论：本轮不替换496。全部预检、40K更新/冻结检查、重载复评、数据签名和本地传输哈希验收通过。
- 结果：`outputs/phase497/phase497_20261110/RESULT.md`；代码`src/phase497`；远端权重与恢复记录`/home/scv/shared/phase497_20261110`。


## Phase498 — 纹理证据进入四阶段空间交互（2026-11-10，已完成）

- 基于496，保留无MsRE网络和末端纹理输出；原环形对比与微纹理编码共享给四个32→64零初始化投影，分别进入交叉注意力K/V空间上下文。新增8192参数，无新损失或阈值。
- 官方DINOv3冻结，任务模块重新初始化；CloudSEN8490/535，目标监督0%，单seed42901、batch2、40K；复用496完成结果作对照，不做多seed。
- 40K完成，最终和最优均为40K。496→498地表/云/影/mIoU：89.2964/85.1042/64.9683/79.7896 → 89.5302/85.4969/65.1339/80.0537；差值+0.2338/+0.3926/+0.1656/+0.2640点。
- 云影FN减少25,159，FP增加811；小幅净收益，未达70%。可训练参数1,494,597→1,502,789（+0.55%），同条件GPU前向27.978→28.184ms（+0.74%）。
- 同一40K权重关闭早期路径后云影65.1324、mIoU80.0526；完整模型仅高0.0015/0.0011点。新连接的推理净指标作用很弱，不能把本轮收益直接归因于早期注入；共享编码的联合训练改变优化轨迹只是待验证解释。
- 保留498权重作为候选，不将新增连接认定为已验证的核心结构。不做多seed。训练过程未恢复重启；SSH监测中断不影响远端训练。
- 结构/梯度测试、真实预检、全部25个纹理参数张量更新、冻结VFM、共同初始化、30个批次签名、最终及最佳重载复评、关断评测、代码锁和传输哈希验收均通过。
- 结果：outputs/phase498/phase498_20261110/RESULT.md；完整40K权重SHA：2ff05e424239cea752dffefdeea17a6631a20c35cfcac66966bab9285838f1f0。
- 计划：src/research_plans/phase498.md；代码src/phase498；远端/home/scv/shared/phase498_20261110。

## Phase499 — RGB协同纹理与环形对比（2026-11-10，已完成）

- 回到496，保留全部原路径，新增原分辨率3/9/21窗口的RGB颜色协方差、方差能量与颜色比例，池化后进行环形比较；60维证据经轻量卷积、零初始化投影进入解码特征。新增15184参数，无498早期连接，不改损失或阈值。
- 官方DINOv3冻结，任务模块重新初始化；CloudSEN8490/535，目标监督0%，单seed42901、batch2、40K。复用496对照，不做多seed。
- 完整40K及重载复评完成，最佳mIoU也是40K。地表/云/影/mIoU=89.6007/85.5904/64.9959/80.0623，相对496为+0.3043/+0.4862/+0.0276/+0.2727点。云影基本持平，主要收益在地表和云。
- 相对496，云影漏检增加19,113，误报减少35,720。关闭新增颜色协方差分支后影64.7081、mIoU79.7635；开启提高0.2877/0.2988点，说明该检查点对分支有推理依赖，但不等于相对496的净结构收益。
- 可训练参数1,509,781（+15,184，+1.02%）；同条件GPU前向28.793→29.895ms（+3.83%）。完整训练及主复评约6404秒，单seed，无中断恢复。
- 合成同灰度/异颜色输入、边界/平坦数值、初始化与梯度、真实预检、全部25个纹理参数张量更新、冻结VFM、30个批次签名、40K重载/关断评测、训练与后处理代码锁及本地传输哈希验收通过。
- 判断：保留结果作记录，暂不把499作为提升云影的主方向；496继续作结构基线。499影64.9959低于498的65.1339，不以整体mIoU略高替代云影目标。
- 结果outputs/phase499/phase499_20261110/RESULT.md；40K权重SHA：7d5291b98ea6aa3e48892b6485e8075a15884efae6ef2b2188d563979e75a593。
- 计划src/research_plans/phase499.md；代码src/phase499；远端/home/scv/shared/phase499_20261110。

## Phase500 — 1/2分辨率边界两侧联合解码（2026-11-11，已完成）

- 496原始535图诊断完全复现：云影FN=2,684,509，FP=2,524,600。72.89%漏检在真值边界内8像素，该区占影像素58.78%；约50.63%误报距云影超过8像素。不把图像外沿当成真值云影边界，ignore邻域另列。
- 保留496，读取空间CNN真实1/2细节，结合掩膜语义学习4个方向、2/6/12原图像素距离的两侧读取及选择，零初始化残差进入128维1/2掩膜特征，由原查询共同生成三类掩膜。新增7813参数，不改损失或阈值。
- 官方DINOv3冻结，任务模块全新初始化；CloudSEN8490训练/535验证，目标监督0%；单seed42901、batch2、lr1e-4、40K，不做多seed。共同层初始化保持，初始全模型预测因掩膜分辨率变化不保证与496相同。
- 完整40K及重载复评完成，最优mIoU即40K。地表/云/影/mIoU=89.4718/85.2769/65.4091/80.0526；相对496为+0.1755/+0.1727/+0.4408/+0.2630点，三类均提高，云影仍未达到70%。
- 云影FN减少28,832，FP减少56,138。≤3像素边界内FN减少29,796；3–8像素FN增加645，内部>8像素增加319。FP在三个距离带分别减少18,056/5,845/32,237。漏检净改善几乎全部来自近边界，内部漏检未改善。
- 只关闭成对读取：影65.3570、mIoU80.0123，完整网络分别高0.0522/0.0403点。关闭整个新增细节残差、保留双线性1/2掩膜特征：影64.8414、mIoU79.2582，完整网络分别高0.5678/0.7944点。推理关断支持关注高分辨率细节路径，但不能单独证明分辨率或中心支路的重训收益。
- 新增7813参数，可训练总数1,502,410；同条件GPU前向29.896→41.310ms（+38.18%）。训练及主复评约8145秒，无中断恢复。
- 结构/采样/梯度、真实预检、19个新增参数及21个原纹理参数张量更新、冻结VFM、共同初始化、30个抽查批次、最终/最优重载、两项关断和边界复测、代码锁及本地传输哈希验收均通过。
- 判断：保留500作为源域云影的新候选；方向有小幅正收益，复杂成对读取的额外推理收益较小，后续优先做简化高分辨率细节结构的对照，不把本轮称为成对机制已被充分证明。
- 结果outputs/phase500/phase500_20261111/RESULT.md；40K权重SHA：aa9487d1049c9631d75aef452a32dd7ef11f2b32e0af1622f70a8b88939f6b1e。
- 计划src/research_plans/phase500.md；代码src/phase500；结果目录outputs/phase500/phase500_20261111；远端/home/scv/shared/phase500_20261111。

## Phase501 — 去掉成对读取，保留高分辨率细节（2026-11-11，训练中）

- 基于500删除方向预测、两侧采样和选择，仅保留1/2分辨率CNN细节、语义与RGB融合及掩膜特征残差；新增6160参数，比500减少1653。
- 官方DINOv3冻结，任务模块全新初始化，CloudSEN8490训练/535验证，目标监督0%；seed42901、batch2、lr1e-4，完整40K。
- 保留模块初始化对齐500；折叠移除通道的常数偏置，验证与500成对输入置零的初始函数一致。真实10步预检及全部11个新增参数张量更新通过。
- 结束后比较496/500/501固定40K指标、云影FN/FP与边界分布、细节分支关断和推理耗时。
- 代码src/phase501；计划src/research_plans/phase501.md；远端/home/scv/shared/phase501_20261111。
