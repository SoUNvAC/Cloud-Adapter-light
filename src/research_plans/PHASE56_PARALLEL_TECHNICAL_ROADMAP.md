# Phase 56–59：类别稳定的跨域 PEFT 并行技术路线

## 0. 任务目标与唯一中心命题

本轮不以“修回 Shadow IoU”为论文命题，也不再重复 Phase 46–49 的因素化输出校准。
需要验证的中心命题只有一个：

> 在四类平坦 softmax 中，少量目标域 thin 监督会通过共享 readout 产生类别补偿性塌缩；
> 将标签本体改写为条件决策，并隔离各条件节点的参数更新，可以在保持 thin 适配收益的同时恢复 shadow。

当前证据锚点（全部按 0–100 报告）：

| 模型 | Thin IoU | Shadow IoU | 弱类均值 | Source-val mIoU |
|---|---:|---:|---:|---:|
| source-only | 11.7114 | 26.7183 | 19.2148 | 73.980 |
| Phase 52A-fix | 37.4599 | 1.0008 | 19.2303 | 73.980 |

Phase 55 已否定“中间表征被抹除/全局梯度对抗”解释：四层梯度余弦均值为
`+0.1997`，第 11 层 shadow probe AUROC 从 `0.5958` 上升至 `0.6383`，聚合梯度
余弦为 `+0.4337`。后续禁止再以表征正交、梯度投影或 shadow 子空间保护作为主方法。

## 1. 与旧路线的边界

Phase 46–49 已覆盖：

- 从四类 logits 做三因素合法重构；
- 从 Mask2Former pixel feature 预测因素残差；
- 源/目标离线因素原型；
- 源锚约束的无标签目标因素 adapter。

这些路线在目标域均未获得收益，禁止通过修改温度、损失权重、训练轮数或原型混合比重启。

特别注意：Phase 46 使用的是 `shadow -> non-shadow` 根节点，再在 non-shadow 内判断
`cloud -> clear`。因此 thin 像素的损失仍包含 `-log(1-P(shadow))`，仍会持续压低
shadow。新实验必须把 `cloud -> non-cloud` 放在根节点；这不是旧实现的复跑。

本轮与旧路线的本质区别：

1. 使用 Phase 52 的同一 1% 目标标注与同一 source replay，不做纯 UDA；
2. 研究的是训练时的条件梯度路径，而不是推理后的概率重构；
3. 每个物理节点有独立可训练参数，thin/thick 损失不得更新 shadow/clear 条件分支；
4. 必须与参数量匹配的平坦分类头、类别平衡分类头和随机标签树比较。

## 2. 单 GPU 下的并行组织

RTX 4090D 只有一个训练资源，所谓并行必须拆成：

- GPU 队列：严格串行，只运行登记后的训练或统一特征抽取；
- CPU-A：数据清单、标签统计、共现统计、随机树生成；
- CPU-B：基于缓存特征训练线性/浅层 readout；
- CPU-C：结果聚合、bootstrap、绘图和报告；
- 开发线：不同人员在独立分支实现，合并前只跑单元测试，不私占 GPU。

定义一个 GPU 预算单位 `1U`：一次 Phase 52A 相同分辨率、相同迭代数的完整训练。
所有候选先跑 `0.25U` 筛选；未过门者不得申请 `1U`。三种子只授予最终一个候选。

建议角色：

| 角色 | 责任 | 禁止事项 |
|---|---|---|
| A：机制审计 | 特征缓存、margin、probe、head swap | 不改模型结构 |
| B：强基线 | 平坦平衡头、cosine head、随机树 | 不加入新 backbone 模块 |
| C：主方法 | 条件头、节点路由、低秩语义输运 | 未过门前不加几何/prototype |
| D：协议与统计 | manifest、队列、汇总、显著性、封存检查 | 不按 val 结果改门槛 |

## 3. Phase 56：无/低训练机制审计

### 56A：统一特征与 logits 缓存

对 source-only 和 Phase 52A-fix，在以下位置保存逐像素抽样特征：

- backbone blocks `[2, 5, 8, 11]`；
- pixel decoder 输出；
- classifier 前最终 mask/pixel embedding；
- 最终四类 logits。

每类、每图最多固定抽样相同数量像素；记录 image id、scene id、真实类、是否 thin-shadow
共现。缓存使用 FP16 特征、FP32 logits，训练/验证缓存物理分离。

交付：

- `feature_cache_manifest.json`；
- 每个层级的 shape、dtype、类别计数与 SHA256；
- 缓存提取脚本和一次可复现日志。

### 56B：readout accessibility audit

只在 target-train 缓存拟合，在 target-val 缓存评估：

1. source-only 与 Phase 52A-fix 的 state-dict delta，确认哪些 head/query/decoder 参数发生变化；
2. 四类普通 multinomial linear probe；
3. 四类 class-balanced linear probe；
4. shadow-vs-thin、shadow-vs-clear 二分类 probe；
5. cosine classifier；
6. source-head / adapted-head 与 source-feature / adapted-feature 的可行交叉组合；
7. GT shadow 像素上的 `z_shadow-z_thin`、`z_shadow-z_clear` 分布；
8. Mask2Former query 归因，区分 mask localization 丢失与 query class assignment 错误。

必须同时报告 AUROC、IoU、召回、ECE 和 margin 分位数。不能用 shadow-vs-all AUROC 代替
四类像素预测。

#### 56B 继续门

冻结 Phase 52A 特征后，最强参数受限 readout 必须同时满足：

- Shadow IoU 至少恢复丢失量的 50%，即 `>=13.86`；
- Thin IoU 至少保留原增益的 80%，即 `>=32.31`；
- 相对 Phase 52A-fix target mIoU 至少 `+2.0`；
- 训练数据只来自 target-train，target-val 不参与拟合或阈值选择。

未通过：终止 readout 主线。结论改为“probe 可检测信息不等于四类可恢复信息”，不得开发层次方法。

### 56C：共现与标签支持审计

分别统计：

- shadow-only、thin-only、thin+shadow、两者皆无的图块数与像元数；
- 65 张主动标注样本中四类覆盖、scene/biome 覆盖；
- 每个 batch 中 shadow 与 thin 梯度实际被观察到的频率；
- shadow collapse 在共现与不共现场景中的条件 IoU/recall。

如果训练集中可用 thin+shadow 共现仍只有 5 张，不允许把共现梯度统计写成总体规律；后续采用
scene-level 分层采样或补标，不能复制这 5 张制造伪样本量。

## 4. Phase 57：标签本体的因果筛选

### 57A：四个参数匹配 head

冻结同一个 Phase 52A backbone/MsRE，使用完全相同的 1% target 标签和 source replay，训练：

| ID | Head | 用途 |
|---|---|---|
| F0 | 原四类 flat softmax 重训 | head-only 基线 |
| F1 | class-balanced flat softmax | 排除纯类别不平衡解释 |
| H0 | 正确条件树 | 检验物理标签本体 |
| O0 | Phase 46 的 shadow-root 树 | 同因素、同参数、同初值的拓扑对照 |
| R0 | 参数匹配随机树 | 排除额外参数/深监督解释 |

正确条件树：

```text
root: cloud / non-cloud
  cloud:     thin / thick
  non-cloud: shadow / clear
```

概率组合：

```text
P(thin)   = P(cloud)     * P(thin | cloud)
P(thick)  = P(cloud)     * P(thick | cloud)
P(shadow) = P(noncloud)  * P(shadow | noncloud)
P(clear)  = P(noncloud)  * P(clear | noncloud)
```

若源四类 logits 顺序为 `[clear, thick, thin, shadow]`，可使用以下零残差等价初始化：

```text
a_cloud^0  = LSE(z_thick, z_thin) - LSE(z_clear, z_shadow)
a_opacity^0 = z_thin - z_thick
a_shadow^0  = z_shadow - z_clear

q = sigmoid(a_cloud)
r = sigmoid(a_opacity)
s = sigmoid(a_shadow)
```

该初始化必须在 FP32 下证明与原四类 softmax 概率逐像素等价，再测试混合精度误差。
H0 的真正自变量是“决策树拓扑决定的梯度路径”，而不是因素数量、参数量或初始化。

条件损失必须使用 mask：cloud 像素才计算 thin/thick，non-cloud 像素才计算 shadow/clear。
实现单元测试需要证明：

- thin/thick 条件损失对 shadow/clear 分支参数梯度严格为零；
- shadow/clear 条件损失对 thin/thick 分支参数梯度严格为零；
- 概率和误差 `<=1e-6`；
- 零残差初始化相对原模型 `max_abs<=1e-5`，若使用精确重参数化。

离线缓存实验应枚举四个带标签叶节点的全部 15 棵满二叉树；在线训练至少保留 O0、
正确树和离线表现最强的错误树。所有树保持参数量和优化预算一致。典型错误树例如：

- `{thin, clear}` vs `{thick, shadow}`；
- `{thin, shadow}` vs `{thick, clear}`。

### 57A 筛选门

单 seed、`0.25U` 预算。H0 必须同时满足：

- 相对 F1：Shadow IoU `>=+3.0`；
- 相对 F1：Thin IoU 下降 `<=1.0`；
- 相对 O0 和最佳随机树：弱类均值 `>=+2.0`；
- source-val 遗忘 `<=1.0`；
- trainable params 不超过 F1 的 `1.10x`，或另给严格参数匹配结果。

未通过：标签本体不是主要因果因素，终止该论文主线。禁止在 H0 上追加 attention、prototype、边界损失。

### 57B：短程重复

只有 H0 通过后，补 seeds `123/3407` 的 `0.25U`。三 seed 的关键效应必须同号；
Shadow 提升的 95% bootstrap CI 下界必须大于 0。未通过则视为不稳定，不进入方法升级。

## 5. Phase 58：唯一主方法的串行进化

每次只增加一个变量；上一层失败即停止后续层。

### M0：条件梯度隔离 head

即通过 Phase 57 的 H0。它是方法最小体，不再重复 Phase 46–49 的推理后校准。

### M1：节点特定低秩语义输运

对三个决策节点分别学习低秩残差：

```text
u_n = W_n_source^T (I + U_n V_n^T) z
n in {cloudness, opacity, shadow_state}
```

约束：

- shared feature `z` 不复制；
- 每个节点的 `U_n,V_n` 只接收本节点条件损失；
- 主对照的总 trainable params 控制在 Phase 52A 的 `<=380,577`；仅探索性筛选允许
  `<=0.60M`，但超过 380,577 的结果不能作为参数效率主结论；
- rank 只允许一次预注册筛选，如 `{2, 4, 8}`，不可逐结果追参。

M1 相对 M0 的继续门：target mIoU `>=+1.0`，弱类均值 `>=+2.0`，source 遗忘
`<=1.0`，且参数匹配 flat low-rank baseline 未获得同等提升。

### M2：源语义锚定

只在 M1 出现 source/readout 漂移时加入。锚定对象为节点 logit margin，而不是全特征 MSE：

```text
L_anchor = sum_n E_source[ |m_n_adapt - m_n_source| ]
```

必须报告有/无 source replay、有/无 margin anchor 的 2x2 消融。若普通 source replay 已等效，
margin anchor 不得作为创新点。

### M3：云影几何关系（可选增强，不是核心创新）

只有前述方法已通过且数据具有可信太阳/传感器元数据时才能加入。若元数据缺失或坐标变换不可靠，
本项删除。几何约束必须与简单方向增强对比，不能把成熟物理知识单独宣称为创新。

## 6. Phase 59：论文级验证

### 6.1 必需域组合

先完成当前 `CloudSEN12 -> Landsat-8 Biome`。主方法冻结后至少增加两个不用于选超参数的域组合：

1. 处理级域移：CloudSEN L1C -> L2A（需 scene-disjoint，防同景泄漏）；
2. 第二传感器/数据集组合：优先选择同时具有 clear/thin/thick/shadow 标签的数据；若标签体系不同，
   必须预注册映射，不得为了结果临时合并类别。

二分类 HRC/GF 只能作为可迁移性补充，不能证明四类条件本体机制。

### 6.2 必需基线

- source-only；
- head-only；
- full fine-tuning；
- LoRA；
-普通 adapter / 项目现有 PEFT；
- MsRE / Phase 52A-fix；
- class-balanced flat head；
- cosine classifier 或 logit adjustment；
- 参数匹配随机树；
- M0、M1、最终方法。

若时间不足，先减少 backbone 数量，不能删除 F1 和随机树这两个关键证伪基线。

### 6.3 统一指标

- mIoU 与逐类 IoU；
- worst-class IoU；
- thin-shadow arithmetic/harmonic mean；
- shadow recall 与 thin/shadow Boundary F1；
- source retention；
- ECE、NLL 和两组关键 margin；
- trainable/total parameters、训练显存、训练时间、推理延迟；
- 三 seed 均值、样本标准差、paired bootstrap CI。

弱类调和均值定义为：

```text
H_weak = 2 * IoU_thin * IoU_shadow / (IoU_thin + IoU_shadow)
```

Phase 52A-fix 的弱类算术均值几乎不变，但 `H_weak` 会因 shadow 塌缩显著下降；后续不得
只用算术均值掩盖单类归零。

最终候选的最低内部目标：

- Thin IoU `>=34`；
- Shadow IoU `>=20`，理想目标为恢复到 source-only 的 `26.7183`；
- source-val mIoU `>=72.98`；
- target mIoU 相对 Phase 52A-fix 有稳定正提升；
- trainable params 主结果 `<=380,577`；
- 所有三个 seed 的 thin 与 shadow 方向一致。

## 7. GPU 队列优先级

```text
P0  56A 一次性特征/logit缓存
P1  56B 缓存上的CPU readout审计
P2  57A: F0/F1/H0/R0/R1，每个0.25U
P3  57B: H0两个补充seed，每个0.25U
P4  M1 rank筛选，最多三个0.25U
P5  最终M0/M1完整1U三seed
P6  第二/第三域组合
P7  可选几何增强与扩展backbone
```

GPU 管理规则：

- 一个中央 `run_registry.csv` 分配唯一 run id；
- 开跑前写入 commit、config hash、checkpoint hash、manifest hash；
- 同时只允许一个训练进程；
- OOM 只能调整 batch/accumulation 保持有效 batch，不得顺便改学习率；
- 失败 run 也保留日志；
- target-test 在最终方法和超参数冻结前保持封存。

## 8. 统一 run id 与交付格式

Run ID：

```text
P{phase}_{route}_{model}_{domain}_{budget}_s{seed}
```

例：

```text
P57_H0_MSRE_CS2L8_q25_s42
```

每个任务必须交付：

1. config；
2. git commit；
3. 完整 stdout/stderr；
4. best/final checkpoint 的选择规则；
5. `summary.json`；
6. confusion matrix；
7. source-val 与 target-val 完整指标；
8. trainable parameter 明细；
9. 是否过门以及机器可检查的失败原因。

`summary.json` 至少包含：

```json
{
  "run_id": "P57_H0_MSRE_CS2L8_q25_s42",
  "commit": "...",
  "config_sha256": "...",
  "manifest_sha256": "...",
  "seed": 42,
  "trainable_params": 0,
  "source_val": {},
  "target_val": {},
  "class_iou": {"clear": 0, "thick": 0, "thin": 0, "shadow": 0},
  "boundary_f1": {},
  "margins": {},
  "gate_pass": false,
  "gate_fail_reasons": []
}
```

## 9. 立即分派清单

### 给 A（今天开始）

- 完成 56A 缓存脚本与缓存 manifest；
- 完成 shadow-vs-thin / shadow-vs-clear probe；
- 输出四类 balanced probe 与 margin 图；
- 不修改任何模型权重。

建议独占文件：

- `src/tools/cache_phase56_readout_features.py`；
- `src/tools/audit_phase56_readout.py`；
- `src/tests/test_phase56_feature_cache_protocol.py`。

### 给 B（与 A 并行开发）

- 实现 F0/F1/cosine head；
- 实现两个参数匹配随机树；
- 写参数量和梯度隔离单元测试；
- 等 A 缓存完成后先在 CPU/小 GPU 上运行。

建议独占文件：

- `src/tools/train_phase56_cached_heads.py`；
- `src/cloud_adapter/models/decode_heads/ontology_heads.py`；
- `src/tests/test_phase57_ontology_gradients.py`。

### 给 C（与 A/B 并行开发）

- 从现有 `FactorizedEncoderDecoder` 提取概率组合与标签 mask 逻辑；
- 新实现必须接 Phase 52 adapted feature 并在训练时隔离条件分支；
- 不复用 Phase 46–49 的无标签原型和最终-logit residual；
- 先交 M0，不提前实现 M2/M3。

建议独占文件：

- `src/cloud_adapter/models/segmentors/ontology_routed_encoder_decoder.py`；
- `src/configs/protocol/phase57_cloudroot_msre_l8.py`；
- `src/tools/run_phase57_ontology_screen_4090d.sh`。

### 给 D（立即执行）

- 建立 `run_registry.csv` 和 GPU 队列；
- 冻结 source/target manifest 及哈希；
- 建立自动汇总与门槛判定；
- 检查 target-test、CloudSEN internal test 未被读取。

建议独占文件：

- `src/tools/phase56_run_registry.py`；
- `src/tools/summarize_phase56_matrix.py`；
- `work_dirs/phase56_protocol/run_registry.csv`（运行产物，不提交大 checkpoint）。

## 10. 总止损规则

出现以下任一情形，终止这条 TGRS 主线：

1. 56B 的受限 readout 无法从冻结特征恢复至少一半 shadow 损失；
2. 正确物理树不能击败 class-balanced flat head；
3. 正确物理树不能击败参数匹配随机树；
4. 效果只在含 5 张共现样本的小子集上成立；
5. 三 seed 效果方向不一致；
6. 需要读取 target-val 调阈值或逐域调超参数才能成立；
7. 最终收益来自更多参数、更多标注或更多训练预算，而非条件梯度机制。

只有通过 56B、57A、57B 三道门，才允许撰写“类别补偿性塌缩 + 本体条件 PEFT”的论文故事。
