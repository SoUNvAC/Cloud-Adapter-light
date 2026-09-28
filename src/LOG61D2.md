# Phase 61D2 实验记录

## 2026-09-26：校准协议与执行前审计

- 目标：重做校准后的盲法复核；reviewer 对原标签、模型和方法盲态，但获得 B1–B11、推荐合成、太阳几何和必要元数据；40 个校准单元不进入统计，160 个独立单元与 50 个未见确认单元进入主复核。
- 改动：新增 61D2 复核包生成、校准锁、分歧专用第三方裁决和评分脚本；支持 `uncertain`、`unobservable_nodata` 与最多两个语义类的集合标签；硬门按 κ/AC1、Thin/Shadow reviewer IoU 和原标签—裁决标签 IoU 自动执行。
- 网络/读出：不训练、不修改任何模型；只做确定性抽样、11 波段证据页渲染和人工复核统计。确认集不按模型错误或置信度筛选。
- 数据口径：旧 200 个错配分层单元中确定性抽取 40 个校准单元，剩余 160 个作为独立复核；另从旧包未出现的 `Shadows?=yes` target-val 图块按 Thin/Shadow、scene/biome、boundary/interior 平衡抽取 50 个确认单元。校准集被评分器结构性排除。
- 止损线：校准锁定前不得开始主复核；A/B 原始判断必须保留，第三位专家只裁决真实分歧；κ 或 AC1 任一低于 0.40 时四类 mIoU 不得作为主指标；Thin 或 Shadow reviewer IoU 任一低于 0.30 时终止四类硬标签主线；不得以旧 RGB 图或 Phase54 三个辅助光谱通道冒充 B1–B11。
- 真实状态：本地 4 项协议单元测试、语法检查和 11 波段证据页合成渲染 smoke test 通过。远端仓库内 `data` 为指向 `/home/scv/shared/data` 的符号链接；完整 B1–B11/MTL 不在已授权的 `/home/scv/Cloud-Adapter-light` 实体目录内，现有 Phase54 cache 也不含全部波段。因此尚未读取外部 raw-root、尚未生成正式复核包、尚无人工结果或 κ/AC1/IoU；等待明确只读授权或将原始数据放入授权目录。

## 2026-09-26：只读授权后生成正式复核包

- 目标：在不泄露原标签、模型输出、方法或抽样来源的前提下，为两位 reviewer 提供完成校准和独立复核所需的完整多光谱证据。
- 改动：用户明确授权只读访问 `/home/scv/shared/data/l8_biome_raw`；生成过程只写 `/home/scv/Cloud-Adapter-light/src/work_dirs/phase61d2_calibrated_review`。原始目录含 96 个 scene 主栈，每个主栈 11 波段，未尝试写入。
- 网络/同步：本地提交 `96759c5` 已 push；gzs 最初连续出现 GitHub GnuTLS/SSH 连接失败，本轮重试后成功 fast-forward 至同一提交。远端使用 `cloud-lite-pt210`，4 项协议测试通过；不训练或修改模型。
- 数据与产物：生成 40 个校准单元、160 个独立单元和 50 个确认单元，共 250 张证据页；确认集使用 50 张互不重复且未出现在旧 61D 包中的图块，原始 Thin/Shadow 各 25。主复核 210 个单元的 biome 数为 snow_ice 44、shrubland 22、grass_crops 40、barren 36、wetlands 38、forest 30。A/B 的校准和主复核 CSV 均保持全空白。
- 质量核验：250/250 PNG 均为 1040×1210 RGB，ID 与 sealed manifest 一一对应；三张跨校准/主复核抽样证据页目视通过；ZIP 有 257 个 reviewer 可见成员和 250 张 PNG，不含 sealed manifest。ZIP SHA256 `df772757afedebab0a82599d59a10ad51bbd2e7b3858721f4862cf0fcb04af49`；sealed manifest SHA256 `7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77`；packet summary SHA256 `0c2f36c49ceeea87b43549d83c0677138adfdd6714b3edc37fe965a4873ea402`；manual SHA256 `cfa8ec5b65474dc4e6d4402cc9db0838db237afd715f081ff88f41cdb068d172`。
- 止损线与结果：状态为 `awaiting_calibration_lock`，`human_results_available=false`。40 个校准单元完成共同讨论并锁定前不得开始主复核；当前没有人工标签、κ、AC1、reviewer IoU、原标签—裁决标签 IoU或四类主线结论，严禁提前填报。

## 2026-09-28：校准文件封存与锁定

- 目标：核验两位 reviewer 的校准提交并锁定统一协议，校准单元不得进入主统计。
- 改动：将本地提交的 `reviewer_A_calibration.csv`、`reviewer_B_calibration.csv`、`calibration_consensus.csv` 原样复制到远端仓库 `completed_reviews/`，未覆盖 reviewer 模板；校准锁脚本同步更新 packet 状态。因共识文件未提供文字版 `rule_or_counterexample`，锁中如实记录“40 个共识标签已提交、无文字规则修订，手册 v1.0 保持不变”，不补写讨论内容。
- 网络/读出：不训练、不推理、不读取 sealed 原标签用于校准；本地提交 `ed9b2a8` push 后，gzs 成功 fast-forward，并在 `cloud-lite-pt210` 中执行锁定。
- 完整性：A/B 校准与共识均为 40/40、ID 集相同、无重复、无空标签、标签语法合法。SHA256：A `f547fdb08d2cd6379fcbc3a61399287919f093d342224d75962ce8482e55eeb2`，B `ff14ad31d411e27ac3150d3b8feadc3c808936aaca30fda05c48b6476aa71284`，共识 `8da009f62145dc4b0c36b3416902875d84f9618c3d7b03bbad26de80385b694d`，校准锁 `0ac14e4d9b384b19a1205cd6e78e050949c126d21884127beecb0c40622fb76f`。
- 止损线与真实状态：40 个校准单元已永久排除最终统计；状态为 `calibration_locked_awaiting_independent_reviews`。A/B 的两个 main CSV 均为 210 行但标签 0/210，仍是空白模板，故 `main_reviews_complete=false`、`human_results_available=false`；未计算 κ、AC1、IoU，未生成第三方分歧裁决，不得宣称61D2已完成。

## 2026-09-28：单 reviewer 主复核降级审计

- 目标：在人力不足、仅 reviewer A 完成主复核的现实约束下，保留人工观察，但禁止把单人判断伪装为独立一致性或金标准。
- 改动：新增单 reviewer 审计脚本，强制校验 210 个 main ID 与标签语法，结构性禁用 Cohen κ、Gwet AC1、reviewer IoU、第三方裁决共识、标注过程偏移门和“回到模型问题”门；packet 状态改为 `single_review_complete_insufficient_for_interrater_inference`。
- 人工限定：reviewer 明确说明“标注本身充满不确定性，黄框内可能同时含有多个语义，引用时需慎重，不可百分比相信”。该说明原文写入审计产物。CSV 中集合标签为 0、notes 为 0；仅 13 个 `boundary_mixed`、2 个 `uncertain`、9 个 `unobservable_nodata`，合计 24/210（11.43%）显式非硬标签。这只能视为被文件编码的不确定性下限，不能据此认定其余 186 项确定无歧义。
- 网络/读出：不训练、不推理；本地提交 `2aaf2ff` push 后由 gzs fast-forward，在 `cloud-lite-pt210` 中运行。reviewer A 文件 210/210 完整、无空值/重复/非法标签，SHA256 `9c00b9349862d7604884507955c64ddffe5eacab81983000016f2827eaf2fcda`；reviewer B 未完成且未作任何代填。
- 探索性描述：A 的 clear/cloud-shadow/thin-cloud/haze-cirrus/boundary-mixed/terrain-water-shadow/thick-cloud/uncertain/unobservable-nodata 计数为 41/66/37/27/13/9/6/2/9。原标签与单 reviewer 的 Thin/Shadow membership IoU 为 25.23%/42.06%（Thin 交27并107，Shadow 交53并126）；这些数值不是 reviewer 一致性、数据集错误率或共识标签性能，只可用于提出假设。
- 止损线与结论：`four_class_miou_may_be_primary_from_61d2=false`，原因是独立 reviewer 可重复性没有被测量。61D2 不能支持“标注过程偏移”或“回到模型问题”的核心结论；允许用途仅为带 reviewer 限定语的描述性、假设生成证据。单评审计 SHA256 `add477f6ede118c8221021e39b5d9ed4cb3c4d7672bb0f370c130df8980f907e`，packet summary SHA256 `5673d60f4ba0939e2e6ef4ab875abc39cb937fcffa9b87176bc9ccc15373d54d`。

## 2026-09-28：双 reviewer 主复核检查与待裁决状态

- 目标：导入 reviewer B 的尽力复核，恢复可计算的双人一致性端点，并只把真实分歧交给第三位专家。
- 改动：B 原始 CSV 原样封存，SHA256 `2f3855752e3d9c0217d2a66c830e184dadee1705d65b83249a252b8b87645afd`。其中 207/210 有显式标签；依据用户既有定义“空值表示纯色、无有效信息、无法评估”，在独立派生副本中仅将 3 个空值映射为 `unobservable_nodata`，不覆盖原件；派生 SHA256 `a4166d07bcb03237d956d4c661f88a0483b60d2d0b63de5aba2e6fcb43bca525`，provenance 保留三个 tile ID、输入/输出哈希与原因。
- 网络/读出：不训练、不推理；本地提交 `b8a4db4` push 后由 gzs fast-forward，在 `cloud-lite-pt210` 中通过 6 项测试并运行双评评分。A/B 均无集合标签和 notes，reviewer 关于多语义与低置信度的自述继续作为所有结果的强制限定。
- 总体结果：210 项 exact/compatible agreement 均为 62.38%，Cohen κ=0.5314，Gwet AC1=0.5820；Thin/Shadow reviewer membership IoU=36.73%/72.73%。全局 κ/AC1 均高于 0.40，Thin/Shadow IoU 均高于 0.30，故仅“总体 reviewer 可重复性最低门”通过；这不是四类主指标已获授权。
- 分层结果：旧错配分层160项的 κ/AC1=0.4828/0.5497，Thin/Shadow IoU=32.43%/72.73%；未见确认集50项为0.6635/0.6877、50.00%/72.73%。biome 差异明显，snow_ice 的 κ/AC1=0.2977/0.3297、Thin/Shadow IoU=0%/25%，其自身未过门；forest 的 Thin IoU=0% 但 union 仅1，不作稳定比例解释。
- 分歧与止损：79/210 项需裁决，其中原分层 boundary/interior 为44/35；分歧类型含 boundary 16、clear↔cloud-shadow 5、cloud-shadow↔terrain/water-shadow 5、thin↔shadow 3、thin↔thick 6、unobservable 8、uncertain 1、其他35。第三方模板已生成但仍为空白，状态固定为 `double_review_complete_awaiting_adjudication`，`formal_human_results_available=false`；原标签—裁决标签 IoU、标注过程偏移门和模型问题门仍不可计算。
- 产物：双评 metrics SHA256 `83be709e461469372952d085b670bdf01b423d31ae9d9dc1695d998414a13fc9`；裁决模板 SHA256 `a28c4201c8ede6ab8ea051b4ee17a5618adde231d275ad99bead61bea5372078`；packet summary SHA256 `55cc57a77ce007dc1c6ab1ea6cb625a4a9d7ddf0e944c707be3f09fff979db54`。
