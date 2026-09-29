# Phase 63 — 可辨识性感知的选择性多粒度域适配可行性审计

## 命题与范围

本阶段只检验一个命题：跨域细粒度云分割的主要风险是否能由现有模型和可观测性证据识别，使系统只在证据充分的区域输出四类细标签，并在其余区域退回预先定义的粗粒度标签或拒识。Phase 63 不训练新分割网络，不选择新 checkpoint，不读取封存测试集，也不把 MsRE 当作核心创新。任何深层 risk MLP、注意力模块或事后阈值修补均不属于本阶段。

## 63A：冻结数据、人工端点和模型

- Phase 61D2 的 40 个 calibration 单元继续永久排除。
- 210 个主复核点全部冻结：160 个 `independent` 单元只能开发风险分数，50 个 `confirmation` 单元只能做一次最终验证。
- 每个单元的人工判断对象是黄色角标中心像元；128×128 crop 只提供上下文。因此本阶段报告 point-level correctness/error，不得称为 dense mIoU。
- 最终标签由 131 个 A/B 一致判断和 79 个第三专家分歧裁决组成。第三专家 79 条理由均保持缺失，禁止补写、推断或据图倒填。
- 210 个最终标签中，明确四类标签仅 162 个：clear 39、thin 42、thick 10、cloud-shadow 71。其余 haze/cirrus 19、terrain/water-shadow 12、nodata 12、boundary 4、uncertain 1 不得强制映射成四类。160/50 中明确四类分别为 126/36。
- snow/ice 必须原样保留并单列。不得因其一致性或性能较低而删除、重分层或事后改阈值。
- scene 是独立抽样单位。同一 scene 不得同时出现在 development 和 confirmation；scene-lock 校验失败时禁止导出确认集指标，先停止并报告冲突，不能静默按点随机划分。
- sealed manifest SHA256：`7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77`；最终人工 metrics SHA256：`929a59fe212921879c87c08e7007404f37ca69f825c5a30f8b90ae5fc9338be9`；第三方原始裁决 SHA256：`90fa08860ea01025b5c750db8b4506f65338e6c3f039017ba8e9b43e9231dce4`。
- 冻结模型对沿用 Phase 55：source-only 为 Phase 22 seed42 iter40000，checkpoint SHA256 `d33a81337544e294a42007c6b353ee13c901ebb1e1f5261fd1d95a79e3888012`；adapted 为 Phase 52A-fix seed52 iter1000，checkpoint SHA256 `9655ee83c63840ecb31351efef0582676529c318fd0d1eca5fd0929db702eb2e`。Phase 62 已按预注册门停止，不能替换该模型对。

## 63B：只读证据导出

在同一中心像元及固定局部邻域导出以下特征，不得查看 confirmation 标签来选择、删除或变换特征：

1. adapted softmax entropy、top-1/top-2 margin和最大概率；
2. 预先固定的 TTA/尺度预测分歧；若某变换不能严格回投同一像元，标为不可用而不是近似填值；
3. source-only 与 adapted 的概率距离、top-1分歧及粗粒度分歧；
4. 仅当存在独立 coarse 输出时计算层级冲突。单个四类 softmax 中 `P(cloud)=P(thin)+P(thick)` 是恒等式，禁止把恒等式报告成层级一致性证据；
5. 到 adapted 预测边界的距离、Thin/Shadow log-odds及二者概率质量；
6. 固定邻域内的 nodata/饱和比例、有效波段数、亮度、局部纹理和缺失模式；
7. biome 作为预注册分类变量，snow/ice 必须有独立输出；
8. 四类预测合并到三类和二类后的预测是否与模型间、TTA间判断矛盾。

原始 logits/probabilities、输入有效性、checkpoint/config/hash、像元坐标、scene、biome、cohort 与全部派生公式必须进入可追溯表。缺少任一模型、坐标不合法、概率非有限或单位ID不匹配时 fail closed。

## 风险端点与允许模型

- primary fine failure：最终人工标签不是可唯一/集合约束到四类的标签，或模型 fine top-1 不属于允许的四类集合。非四类人工标签保留为“不能可靠作四类判断”，不伪装成 clear 或 thin。
- Thin/Shadow coverage 只以最终标签含明确 thin/cloud-shadow membership 的单元为各自分母。
- 三类映射仅对明确四类端点定义为 `clear / cloud(thin+thick) / cloud-shadow`；二类映射为 `cloud(thin+thick) / non-cloud(clear+cloud-shadow)`。非四类人工端点在这些指标中单列为不可评价，不暗中并类。
- `clear/contaminated` 仅作预注册次要端点；nodata、boundary、uncertain 必须不可评价。
- 风险模型仅允许逻辑回归作为 primary；isotonic 只能在 development 内校准单一预注册分数；GAM 仅作明确标记的敏感性分析。超参数和阈值只能用按 scene 分组的 development 内部交叉验证确定，confirmation 不得拟合、校准、选特征、选模型或选覆盖率。

## 63C：一次性生死门

在 scene-lock 通过且所有输入哈希匹配后，锁定风险模型和阈值，再一次性读取 confirmation 标签。选择性路线必须同时满足：

1. confirmation 错误检测 AUROC ≥0.70，scene bootstrap 95% CI 下界 >0.50；
2. fine coverage ≥60% 时，accepted fine error 相对全覆盖基线下降 ≥25%；
3. Thin 与 Shadow 各自 coverage 均 ≥30%；
4. 预注册三类 coarse correctness 相对同一冻结模型的全覆盖合并输出下降不超过1.0个百分点；二类结果同时报告；
5. development 与 confirmation 的 AUROC、选择性错误下降方向一致；
6. snow/ice 原样单列，且总体结论不得通过删除该 biome 获得。

bootstrap 以 scene 为重采样单位；若 confirmation 的 scene 数、正负错误类或 Thin/Shadow 分母不足以形成定义良好的统计量，门按失败处理，不改成 point bootstrap。任一门失败即执行 `stop_selective_multigranularity_route`，禁止据此继续发明 attention、router 或损失模块。

## 并行工作流与合并门

- A组：只实现证据表、风险—覆盖曲线、scene bootstrap和硬门评分。
- B组：只冻结160/50 scene-level划分并核验泄漏；泄漏时 fail closed。
- C组：只实现层级概率合并、冲突读出和外部风险阈值 router 原型；不训练、不注册正式配置。
- D组：寻找第二个真正独立目标域，核对标签兼容性和公开基线；未取得数据与合法可比端点前不得声称复现完成。

A/B/C/D 的输出只有在共同数据哈希、标签语义和scene-lock一致时才能合并。Phase 63 的正式运行必须另行记录目标、改动、网络情况、止损线与真实结果到 `src/LOG63.md`。

## 63A scene冻结实际预检结果：fail closed

真实 sealed manifest 的 scene 审计未通过：development 与 confirmation 都来自同一组8个scene，重叠为8/8。各scene的development/confirmation单元数依次为14/8、14/8、14/8、21/8、32/6、26/4、32/4、7/4；总计仍为160/50，snow/ice仍为28/16，但scene独立性为零。审计产物SHA256为`cbeb56b9b2964110bb5162e23e78127f9ff1469162f1c31e536784ebc9b11ee2`，`phase63_split_lock.json`未生成。

这不是可安全修复的代码错误，而是现有抽样设计与新预注册条件的冲突。在保持现有50项“锁死”不动的前提下，不可能再让同scene的160项用于风险开发。正式63C因此暂停，且不得输出带泄漏的confirmation AUROC或bootstrap门。可接受的后续修复只有在任何风险拟合前另行预注册：优先从此前未见scene取得新的独立确认集；若改为按完整scene重划现有210项，则必须明确废止原160/50角色，且不能把已被分析过的旧确认标签继续称为未触碰确认集。
