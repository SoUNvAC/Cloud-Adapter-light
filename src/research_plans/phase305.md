# Phase305：高置信Surface损伤定位与固定读出诊断

## 一、做什么

复用Phase304封存结果，定位新增Surface→Cloud错误是否发生在Source原先高置信且正确的Surface区域；按参考云边界、场景地表标签和全部32组拆分。用一个固定、无标签可计算的Source优先读出检查潜在保护收益及纠错代价，为后续有向约束提供依据。

零训练、零模型前向、零下载。不重新生成伪标签、不改阈值、不选checkpoint、不回写Phase304结果。本轮完成CPU统计、数值验收和报告。

## 二、怎么做

### 1．固定缓存、支持和类别

- Source为Phase302原Source；学生为Phase304固定Q90-PL末步。引用各自EVALUATION_LOCK、ENDPOINT_LOCK、INPUT_SHA256及完整备份，核验实际文件SHA，不仅按文件名判断身份。
- 本地缓存入口：`outputs/phase302/phase302_20261020/remote_backup/evaluation/Source_<product>.npz` 与 `outputs/phase304/phase304_20261024/remote_backup/evaluation/LoRA_<product>.npz`。以锁中实际文件为准；缺文件先查已校验备份，不补模型推理。
- 仅原Catalogue32个development代表，使用Phase304同一有效索引、common支持、参考one-hot/ignore及三类映射Surface=0、Cloud=1、Shadow=2。不能因缺类、极端损伤或置信度低删组。
- 先复现两策略全部逐组/池化混淆和27格T。固定核验：Surface→Cloud消除27632、新增526351、净增加498719；新增中Source原正确Surface→学生Cloud为506798，Source原错误Shadow→学生Cloud为19553，两者和为526351。其余五种有向错误也逐项复现。
- s为原512九窗拼接后的三类Mask2Former语义得分。仅用Source s计算q=s/sum(s)，sum<=1e-12不进入高置信条件并单列；不得改成softmax(s)或用学生置信度筛选Source。

### 2．先固定不依赖真值的条件，再统计正确性

在原共同image-valid支持上定义：

- `G = (Source_pred==Surface) & (q_source[Surface]>=0.90) & (sum(s_source)>1e-12)`。这是潜在部署可计算条件，不使用标签；0.90沿用既有Q90，不另扫阈值。
- `H = G & (truth==Surface)`，只供离线正确性审计；H不能写成推理时可访问条件。
- `D_correct = (truth==Surface) & (Source_pred==Surface) & (student_pred==Cloud)`，全部506798个；另列 `D_other = (truth==Surface) & (Source_pred==Shadow) & (student_pred==Cloud)`，全部19553个。

将D_correct按G/非G分成互斥两层。报告 `|D_correct∩G|/|D_correct|`（损伤集中比例）和 `|D_correct∩G|/|H|`（高置信正确Surface被损伤的比例），两个分母不得混用。非G层同样列Source原正确Surface支持与损伤率；零分母为NA。

同时统计G的真值Surface/Cloud/Shadow支持，报告G的Surface precision，以及G内Source原错→学生改正的数量，特别是原判Surface但真值Cloud、学生判Cloud的纠正。不能把高置信Source当绝对真值；保护G也可能挡住必要修正。

### 3．固定分层，不按结果选场景

- 参考云边界邻域及半径2排除规则完整复用Phase301/304，划分boundary/nonboundary/excluded三层；三层覆盖原共同支持且互斥。不得丢掉excluded后仍称全体支持。
- 每层列G/H支持、D_correct在G/非G的数量及损伤率；同时保留D_other。所有统计按产品/相关组另交，最大损伤组只是集中度描述，不替代全32组。
- 场景标签仅使用现有 `data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv`，按scene完整产品ID连接。固定字段为forest/jungle、snow/ice、agricultural、urban/developed、coastal、hills/mountains、desert/barren、shrublands/plains、wetland/bog/marsh、open_water、enclosed_water。每字段分别列0/1/缺失，保留零支持；字段互相重叠，不跨字段累加。
- 这些是场景标注，不是像元级地物标签。例如只能称“snow/ice=1场景中的参考Surface错误”，不能称“雪像元被误判云”。不引入难度、annotator、真值类别比例、预测亮度分箱或新的人工判读用于挑组。
- 提交字段映射及来源SHA。产品重复、冲突或缺失不得猜值，明确归缺失层。已有表的地物名称不等于已确认物理致错机制。

### 4．一个固定Source优先读出，只用缓存

固定诊断预测：若G且学生判Cloud，则改为Surface；其他位置完整保留Q90学生预测，称 **Q90+G90读出**。条件中不得使用truth或H，不为每景选择不同规则。

- 统计该读出修复的真值Surface数量、撤销的真实Cloud纠正数量、对真值Shadow的变化；先报计数，再报三类混淆/IoU、组均及池化mIoU、Shadow召回/误报。
- Cloud/Surface互换不会改变Shadow决策，因而所有Shadow TP/FP/FN/TN、默认召回/误报必须与Q90精确相同。沿用原Shadow连续评分，AP同样不变，不能声称该读出提高云影排序。
- Source-only/Q90/Q90+G90三行全部报告。不产生新的得分、阈值或可训练参数；本轮不将读出部署，也不根据表格再改G。
- 这张表只回答固定保护条件的可利用空间与代价，不证明训练时加入margin约束能达到同样效果。线上若同时需要Source和学生预测，存在额外推理成本；本轮缓存计算零前向不代表部署零成本。

### 5．正确性与交付验收

- 逐组Source/学生混淆、27格转移边缘及全部有向错误净变化均精确复现Phase304。
- G/非G拆分与boundary/nonboundary/excluded拆分分别完整；D_correct+D_other=新增Surface→Cloud；修复/新增计数与新读出混淆之差吻合。用独立布尔交集核对主要事件，不只调用主计数函数。
- 对G90读出验证：改动仅限G且学生Cloud的位置，其他逐像元不变；Shadow掩膜和原评分逐像元不变。IoU union=0取NA，其余规则沿用Phase304。
- 每个地表字段的0/1/缺失层相加必须回到全池支持与事件；独立组数不因字段重叠、像元多而增加。
- 保存输入/代码/映射/输出SHA，记录耗时、峰值内存、实际缺项及失败。只写新目录，不覆盖历史缓存/报告。

## 三、交付什么

输出至 `outputs/phase305/phase305_YYYYMMDD/`，摘要至 `src/research_results/phase305_YYYYMMDD/`，日期用实际执行开始的北京时间。`RESULT.md` 两页内，分“数据结果、结论分析”。

**表一：损伤是否落在高置信正确Surface。** 全体及逐组，G和非G均有行；另列D_other，不混入D_correct分母。

| 区域/组 | G或非G | Source预测Surface数 | 其中真值Surface数 | 原正确Surface→学生Cloud数 | 占D_correct比例% | 对应正确Surface损伤率% | D_other数 |
|---|---|---|---|---|---|---|---|

**表二：G内可靠性与纠错代价。** 全体及逐组。

| 区域/组 | G支持 | 真值Surface/Cloud/Shadow数 | Surface precision% | Source原错→学生改正数（分真值类） | G且学生Cloud：真值Surface/Cloud/Shadow数 |
|---|---|---|---|---|---|

**表三：边界与场景地表分层。** 两种分层各自单独交CSV，不混合后相加；场景字段列0/1/缺失。

| 字段/区域 | 取值及定义 | 产品/相关组数 | G/H支持 | D_correct在G/非G数量 | 对应损伤率% | D_other数 | 占全池对应事件比例% |
|---|---|---|---|---|---|---|---|

**表四：固定读出诊断。** Source-only、Q90、Q90+G90三行，以及G90−Q90差值。Shadow指标只引用/复核旧值，不重定义。

| 决策 | 组均/池化mIoU | Surface/Cloud/Shadow IoU | Surface→Cloud数 | Cloud→Surface数 | Shadow TP/FP/FN/TN | 宏召回/误报 | Shadow宏/池化AP |
|---|---|---|---|---|---|---|---|

**表五：固定读出的实际改动。** 全池和全部32组，保留零项。

| 组 | 总改动数 | 修复真Surface误判Cloud | 撤销真Cloud纠正 | 真Shadow的Cloud→Surface变化 | 其余位置是否不变 | Shadow掩膜是否不变 |
|---|---|---|---|---|---|---|

另交完整逐组表、G/H支持与真值分母、分层映射锁、源/学生/读出混淆矩阵、输入/代码SHA、独立CPU验收、实际资源及失败记录。报告直接回答：损伤是否主要位于原高置信正确地表；涉及哪些参考位置和场景标签；固定保护会牺牲多少必要纠错；是否值得下一轮只测试一个有向margin保持约束。

不新增bootstrap/阈值扫描，不声称统计显著、物理因果或云特有新方法；历史development已多次使用、单seed与Source预训练交集限制照实披露。若高置信区域损伤占比低或纠错代价高，报告该保护条件的局限，不为得到好结果调整门限。

资源：网络前向=0、训练=0、下载=0；CPU<=16GiB、运行<=1小时、新持久文件<=1GB、宿主空闲>=4GB。逐产品流式处理，不将全部全景s重复留在内存。缓存缺失时交具体缺项，不补神经推理。无自动push、定时检查或下一phase；本文件只立项，不启动远端运行。

## 执行授权补充（2026-10-25）

用户随后明确“执行305”。沿用本聊天已授权的本地修改、git push、gzs pull及实验定时检查；全部CPU计算在gzs的cloud-lite-pt210，本机仅编辑、文件哈希与备份。计算严格零训练、零前向、零下载；原Phase302/304缓存和报告只读，完成后停止检查，不启动下一phase。
