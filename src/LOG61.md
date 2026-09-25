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

## 2026-09-25：61B 优化充分性审计完成

- 目标：区分优化不足、训练支持不足和验证泛化/标签异质性。
- 改动：冻结 Phase52 `pixel_decoder_mask`，对 36 个方法×预算组合使用同一线性 softmax、class-balanced CE、AdamW、4000 steps、每 step 4 张图；不更新分割网络。
- 网络/采样：有效像素比例 source/target=0/100%；16/32/65/130/325 的平均每图曝光为 1000/500/246.154/123.077/49.231 次。
- 止损线：train Thin 低指向优化/表示能力；train 高而 val 低指向覆盖/异质性；325 未收敛才可归因曝光不公平。
- 结果：所有 325 组均满足 loss 收敛判据，故 325 失败不能归因于曝光不足。Random 的 train→val Thin/Shadow：16 为 50.67/70.98→28.80/51.05，32 为 43.40/63.02→30.68/51.86，65 为 44.04/58.44→30.56/55.28，130 为 42.44/57.68→33.09/54.73，325 为 40.37/57.32→33.33/56.00。
- 关键诊断：原 Phase50 nominal-65/合法12 的 train Thin/Shadow=65.70/67.54，val=21.97/29.97，存在显著 train–val 断层。Random65 并未稳定优于 label-aware；Phase50-rebased@65 的 val Thin/Shadow=32.91/52.40。单 seed 不允许使用“稳定优于”措辞。
- 产物：SHA256 `e4b2da63bd9decc265f89a6d1454c87980296f582e9168985a795af55baff176`。

## 2026-09-25：61C 标签粒度坍缩完成

- 目标：对完全相同的四类 confusion 仅折叠标签 ID，测量四类→三类→二类的性能恢复。
- 改动：无推理重跑、无训练；三类为 clear/cloud(thin+thick)/shadow，二类为 cloud/non-cloud，另报 clear/contaminated。
- 结果（四类/三类/cloud-noncloud/clear-contaminated mIoU）：Phase52 source-only=41.91/58.36/76.39/74.38；Phase52 adapted=43.67/48.92/75.13/71.51；Phase56 readout=43.28/55.24/76.42/71.64；Phase60 random65=47.08/57.33/76.89/70.42；oracle325=47.70/56.91/76.22/70.43。
- 判定：粗粒度任务大幅恢复，支持主要域差异包含 thin/thick/shadow 边界语义；但这只是计算证据，61D 独立复核完成前不得发表“标签体系不一致”的核心结论。
- 产物：SHA256 `9d1c09593d3ed009a30d2fab82dbc0076660eb4d88780f29afd58e0eabf64ee3`。

## 2026-09-25：61D 双盲复核包准备完成

- 目标：为至少两位独立人员提供不知道原始标签、模型结果和抽样来源的复核材料。
- 改动：冻结 Phase52/Phase56 预测，从 `Phase52 pred thin/GT shadow` 与 `Phase56 pred shadow/GT thin` 两类错配中，按 biome、S4 高/低置信度、boundary/interior 分层生成 128×128 RGB crop；生成封存映射、说明、两份独立空白 CSV 和评分脚本。
- 网络/读出：仅冻结推理；不训练或修改任何网络。候选 7,884 个，最终盲化单元 200 个，图像文件 200 个；reviewer_A/B 均为表头加 200 条空白记录。
- 止损线：两份独立 CSV 完整锁定前，Cohen/Fleiss κ、共识 IoU、thin/shadow 一致率、原标签一致率和 biome 一致性均不得报告。
- 结果：状态 `awaiting_two_independent_human_reviews`；`human_agreement_metrics_available=false`。sealed manifest SHA256 `8c774bc2f121459fe780654fbae7178b78e1113446d51b2a7ca6a4ebf74a66fd`；packet summary SHA256 `0d5b6aa87ae5ebc52ac230d662e04ceb567877d56c7f36dfc8d6bfd6d6118f49`；Phase61 总摘要 SHA256 `3018bb09e3c266a464fa696ab4495070088d9fcc1144d31df18a18ad4ed59c05`。
- 判定：计算审计完成，但 Phase61 整体尚未完成；在两位人员复核前，只能说“标签粒度不匹配假说获得计算支持”，不能发表“两个数据集标签体系不一致”的核心结论。

## 2026-09-25：61D 双盲复核首次评分（缺失值临时审计）

- 目标：导入 reviewer A/B 的独立复核结果，计算一致性、类别交并比、thin/shadow 一致率、原标签一致率及 biome 分层稳定性。
- 改动：将 `/mnt/i/zhc/reviewer_A.csv` 与 `reviewer_B.csv` 原样复制到远端 `src/work_dirs/phase61/blind_review/completed_reviews/`，不删除或改写源文件；评分器新增严格缺失校验、complete-case 临时模式、输入 SHA/缺失 tile 留痕，以及唯一无歧义的拼写规范化 `ambiguous haze/cirru`→`ambiguous haze/cirrus`（1 条，不作语义代填）。
- 网络/读出：不训练、不推理、不修改网络；在 `cloud-lite-pt210` 中只运行统计评分。本地提交 `7ef9da8` 已 push；gzs 首次 pull 遇到 GitHub GnuTLS 握手失败，第二次重试成功并 fast-forward 到同一提交。
- 数据完整性：sealed 200 条；A 完成 193、缺 7，B 完成 198、缺 2；双方缺失重叠 1 条，因此双人 complete-case 为 192，排除 8 条。A/B 原始 SHA256 分别为 `d5890697dafa64fe8b4fc77fa244b6ce244ac8be1d45d8be9b2c30b4802d3477`、`931b28ac4c82034b29adb85ad106cb3e357815e8335094448e9f280fd68ae440`。
- 止损线：状态固定为 `incomplete_preliminary`；不得把 192 条临时结果冒充完整 200 条正式结论，不得代填 9 个 reviewer 空单元，也不得据此发表“标签体系不一致”的核心结论。
- 结果（192 个双人有效单元）：Cohen κ=0.1363，Fleiss κ=0.1286，精确一致率=28.65%。reviewer 间 IoU：clear 41.56、thin 8.06、thick 5.88、cloud-shadow 19.35、terrain/water-shadow 0.00、ambiguous-haze/cirrus 7.84、uncertain-boundary 9.84（%）。原标签均为 thin/shadow 的抽样单元上精确一致率同为 28.65%；仅在双方都选 definite thin/shadow 的 17 条中，thin-vs-shadow 一致率为 64.71%。
- 共识/原标签：两人完全一致才形成共识，共 55/192（28.65%）；原标签与共识一致率 12.73%，Thin/Shadow IoU 分别为 14.29/10.53。该错配抽样设计没有 original clear/thick 支持，因此 clear/thick 的原标签 IoU 不作解释。
- biome 一致性（双人精确一致率）：barren 15.00%、forest 33.33%、grass_crops 36.11%、shrubland 27.78%、snow_ice 31.03%、wetlands 30.56%；各 biome 均未显示高一致性，但缺失主要集中在 snow_ice（29/36 complete-case），只能作为临时诊断。
- 产物：`review_metrics_preliminary.json` SHA256 `d24112930f1fa39c8c1b80d9ef5e32a2b3cafee55c08d38ad765c8ab861920db`；结构与状态断言复核通过。
