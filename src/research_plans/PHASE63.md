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

## 63A2：新增独立人工确认集修复协议（风险拟合前预注册）

用户确认可以继续组织人工复核后，废止旧50项作为Phase63最终确认集的角色；旧50项与旧160项一起只属于既有Phase61D2证据，仍不得把同scene拆分结果报告为独立验证。新人工确认集固定为50项、11个scene，并在任何Phase63风险拟合前由确定性规则选出：

- 严格确认队列：从8个未进入旧160项、未进入Phase50所选训练scene的`target_val`且USGS元数据`Shadows?=no` scene中，每scene确定性抽4个patch，共32项；
- snow/ice压力队列：从3个未进入旧160项、且Phase50没有选中任何patch的`target_train` snow/ice scene中，每scene确定性抽6个patch，共18项；这些scene对Phase50适配训练是零梯度暴露，但仍属于训练候选分区，不能伪称严格未见验证集；
- `target_test`继续完全封存，选择过程不得读取它；patch与中心坐标选择不读取像素标签、人工标签、模型预测或风险分数，但按预注册的`Shadows?`和biome协议元数据分层，因此准确表述为“像素标签盲、模型盲的分层抽样”；
- 32项严格确认队列和18项snow/ice压力队列必须分别报告。50项合并结果只能作为分层敏感性结果，不能当作单一总体的无偏发生率估计；Phase63C主门以预注册脚本明确的队列为准，禁止看完人工标签后切换；
- 复用原40项校准图，但旧人工判断不进入新reviewer可见包。两位reviewer先独立填写校准表、共同讨论并记录共识，再独立完成新50项；保留原始A/B文件，第三专家只裁决真实分歧；
- reviewer可见每项B1–B11、真彩色/假彩色/指数合成、太阳方位和必要元数据，但看不到原标签、旧人工标签、模型输出、风险分数、抽样来源或方法身份；允许`uncertain`、`unobservable_nodata`与最小集合标签，不强迫四选一。

该修复只建立新的人工端点，不构成63C结果。在两位reviewer与第三专家完成、原始文件哈希冻结、scene-lock再次通过之前，仍禁止拟合最终风险模型、读取新确认标签计算门指标或启动任何新分割训练。

## 63A3：评价单位修正与精确像元 UI pilot

对新校准记录的复查发现，旧证据页虽写明“bracketed pixel”，但黄色角标在源分辨率上实际包围约20×20像元，不能唯一指定一个像元。旧A/B低一致率因此混合了“评审者选择框内哪个区域”的定位歧义与thin/thick/shadow等语义歧义；不得再把它全部解释成细粒度类别不可复现。Phase61D2数值保留，但必须增加该限制说明。

正式confirmation暂停。先运行一个永久排除于模型、风险和最终confirmation统计之外的16项UI pilot：8个biome各选一个此前未人工复核的`target_train` scene，每scene两个未被Phase50选中的patch；这些scene允许有Phase50训练暴露，因为pilot只审计人机界面，不评价模型。抽样不读取像素标签、模型预测或风险分数，`target_test`保持封存。

每页同时展示192×192上下文与21×21最近邻像元视图；上下文角标只定位，局部视图的空心方框严格包围一个源像元，十字线停在方框外，并显示512像元patch内的`(x,y)`坐标。填写顺序固定为：目标是否可定位；可细分类/混合边界/证据不足/nodata；仅当可细分类时填写单一语义类别；记录耗时。边界状态不再与语义类别处于同一竞争标签空间。

UI可行性门在看新判断前锁定：双方目标可定位率均至少95%；可辨识性exact agreement至少70%且AC1至少0.40；双方均认为可细分的项目至少6个，且这些项目语义exact agreement至少50%。新图避免了记忆泄漏，但与旧40项不是同一批难度，故pilot只能判断新界面是否达到继续正式confirmation的最低可行性，不能把前后差异当作UI改动的因果效应。任一门失败则继续暂停confirmation，不得用讨论后共识替代独立复核。

## 63A4：Semantic-Coverage Exact-Pixel UI Pilot（取代63A3随机点）

63A3的16项坐标由无语义哈希生成，只能检查准星定位，不能保证Thin/Shadow等研究对象覆盖；该包在任何reviewer填写前标记为`superseded_before_review`，不得产生结果。63A2的50项正式confirmation中心同样无语义条件，其中32项来自`Shadows?=no` scene，不能保证Thin/Shadow coverage分母；文件保持未读取，但状态改为`suspended_not_formal_confirmation`，不再自动拥有正式确认集资格。

新pilot固定为20项，名称为`Phase63 semantic-coverage exact-pixel UI pilot`。原始标签只在sealed端作为抽样层，不向reviewer展示，也不作为参考真值。抽样与坐标必须满足：

- 名义interior：clear、thin、thick、cloud-shadow各4项；中心像元到任意8邻域标签边界的精确欧氏距离必须`>3 px`，优先`>=5 px`；
- 名义boundary：thin↔thick 2项、cloud(thin或thick)↔shadow 1项、clear↔shadow 1项；中心像元到指定类别对的8邻域边界距离必须`<=1 px`，实现固定选择直接相邻的界面侧像元（距离0）；候选不足立即停止，不得目视换点或替换类别对；
- 优先20个不同scene；若不存在合法解，则在每scene最多2点的约束下最大化不同scene数；四个interior类别各至少覆盖3个scene；同一patch最多1点；
- 精确记录距离、边界两侧标签、scene、patch与缓存哈希，全部只进入sealed manifest；reviewer包不得包含名义标签、抽样层、距离或边界对；
- 只读取已授权L8 Biome原始`fixedmask.TIF`，并限定USGS `Shadows?=yes`的`target_train` scene；原始值固定映射为clear=0、shadow=64、thin=192、thick=255，fill=128保持无效且不得当clear抽样。禁止读取模型特征、概率或预测用于选点；target-test保持封存。

UI与标注流程门固定为：A/B目标可定位率均至少19/20；坐标、页面和提交格式错误为0；双方均可定位项的可辨识性exact agreement至少70%；16个名义interior项中双方均判可细分至少10项；这些项目语义exact agreement至少60%。每个名义层只报告原始计数，不报告每类准确率。两份原始判断必须先校验格式并冻结SHA256，之后才允许解封名义层作描述性`reviewer–nominal-label concordance`；该concordance不是人工准确率。

本20项仅检验UI和标注流程。通过后也只能重新预注册scene独立、模型盲且按sealed名义类别分层的正式confirmation；不得用本pilot证明标签可复现、类别性能或Phase63风险模型有效。旧50项在新正式confirmation协议出现前持续暂停。
