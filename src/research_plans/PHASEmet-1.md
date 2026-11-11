# phasemet-1：场景mIoU分母敏感性审计执行单

版本：2026-10-15。优先级：3。MET = Metric Audit；进度1。独立CPU任务，不依赖CAL/SCD结果。

## 任务范围

读取Phase81和84已经生成的三类混淆矩阵，按两种固定类别纳入规则计算场景mIoU，交付差值、误分计数和排序变化。**不运行任何模型、不读影像、不训练、不改旧指标或历史通过／失败判定。最终只交付 `phasemet-1-result.md` 中的一张小表；下文完整矩阵、逐组明细和检查记录留执行端，不另交付，也不为打包新增报告。**

## 1. 输入文件

仓库 `/home/scv/Cloud-Adapter-light`，相对路径：

```text
outputs/phase81/phase81_20261011/confusion_matrices.json
outputs/phase81/phase81_20261011/development_groups.csv
outputs/phase81/phase81_20261011/summary.csv
outputs/phase81/phase81_20261011/phase81_report.json
outputs/phase81/phase81_20261011/LOCAL_VERIFIED.json
outputs/phase84/phase84_20261011/confusion_matrices.json
outputs/phase84/phase84_20261011/development_groups.csv
outputs/phase84/phase84_20261011/summary.csv
outputs/phase84/phase84_20261011/phase84_report.json
outputs/phase84/phase84_20261011/LOCAL_VERIFIED.json
```

若Linux不存在上述小聚合文件，可通过已有受控交付渠道复制同SHA文件；不复制整个outputs、work_dirs或大缓存。按已有artifact／运输清单绑定原SHA，记录本次输入manifest。缺文件停止，不重跑81/84补数据。

Phase81的组数组字段为 `groups`，Phase84可为 `rows`。解析时分别适配，必须校验每条产品身份、矩阵及原指标，不能假定JSON结构相同。

固定策略：81的MM/LL/LM/ML；84的LL/R4/F真实/F打乱。F策略机器键从输入文件读取并记录与显示名的对应；每个显示名必须唯一，不能凭字典遍历顺序认策略。跨phase的相同LL只作复现交叉检查，不当新增样本。

## 2. 数据检查

1. 从旧报告与计算代码确认矩阵轴和类顺序：预期行是真值，列是预测，Surface/Cloud/Shadow。不能只因矩阵为3×3就默认轴方向。
2. 矩阵必须是非负整数3×3，各行和为参考类像元数，各列和为预测类像元数；总有效像元每组应与原记录1044484相等。
3. 每策略应有原32产品／组，不重复、不漏组。各策略逐组参考行和必须一致；不允许用不同参考支持比较。
4. 各策略逐组矩阵求和必须等于其pooled矩阵。原组等权结果与池化结果分别核验，禁止对各组mIoU按面积平均代替池化。
5. 先精确复现旧定义下的逐类IoU、逐组mIoU和宏／池化摘要。比例域容差≤1e-10；只提供舍入CSV时使用原JSON机器值核验并记录来源。
6. 不一致即暂停并交付差异，不修改原文件“修复”数值。

## 3. 两种固定计算

设混淆矩阵为M，行和 `t_c`、列和 `p_c`、对角 `TP_c`：

```text
union_c = t_c + p_c - TP_c
IoU_c = TP_c / union_c                # union>0时
U = {c : union_c > 0}
T = {c : t_c > 0}
old_mIoU = mean(IoU_c for c in U)
reference_classes_mIoU = mean(IoU_c for c in T)
```

union=0的类别IoU记NA，不填0或1。有效总数>0时T不能为空；空则是输入错误。`reference_classes_mIoU`是新增敏感性指标，不替换old_mIoU。

每组保存：t、p、TP、union、三类IoU、U/T成员、两种mIoU、`missing_reference_predicted_classes=U-T`、这些新增类的伪阳性计数、总错分像元及错分率。

另外直接由原矩阵计算，禁止换定义：

- Surface→Cloud、Cloud→Surface、两者合计及有效像元占比；
- Shadow→Surface、Shadow→Cloud、Shadow召回；
- 非Shadow→Shadow误报计数／全部非Shadow分母；
- 每策略原pooled三类IoU。

这些计数和逐类IoU在两种平均规则之间必须不变。

## 4. 固定比较与表格

只做下列比较，不搜索更有利组合：

|来源|比较|
|---|---|
|81|LM−LL；LM−MM；ML−MM|
|84|F真实−R4；F真实−F打乱；F真实−LL|

每个比较在同一组上报告：两规则各自mIoU差、二者差额、差值符号是否改变（按原始float64严格正／零／负，另保留绝对数值）、有效类数是否改变、实际误分增减。

按**参考有1/2/3类**固定分层，统计组数、两规则宏差、符号改变组数、参考缺席却被预测出的类及像元数；不按效果筛组。

每phase分别为其四策略生成两规则下宏mIoU排序，原始数值同值并列，不靠舍入制造或解除平手。不可将不同phase合起来评选总冠军。

核对并明确列出这些已记录案例，同时保留全32组：

- 81 T20WNV：全Cloud，LM相对LL有254个Surface误分，旧类别数1→2；
- 84 T51TUM：全Surface，真实F新增1292个Cloud误分；
- 84 T50PLR：全Cloud，真实F新增74个Surface误分；
- 84 T44UNA：核对实际计数，不能用“约50pp”反推；
- T32NKF：保留真实Surface/Cloud混淆变化，不归并为缺类分母现象。

不新增bootstrap、显著性判定、模型训练或图表；原有区间不改算后冒充原预注册结果。

## 5. 实现、预算与停止

本轮尚无独立MET执行器。执行工作包括补齐 `src/tools/run_met1_audit.py`，只依赖标准库／既有NumPy，接受显式输入manifest和新的output目录。不能在旧81/84脚本中修改计算规则后覆盖旧报告。

最少CPU合成测试：完美单类、单类加少量异类误报、真值有三类、完全无预测的类、union=0、矩阵转置错误检测、逐组求和与pooled一致。审计主计算与第二实现分别从矩阵直接算，不只比较复制的输出列。

CPU≤4线程、RSS≤2GiB、计划≤30分钟、新增文件≤128MiB；GPU调用／影像读取／训练／下载均0。超预算先停止留回执，不扩池。宿主物理盘保留≥4GB。

脚本没有配套时不执行旧模型补结果；可交付 `blocked_missing_cpu_entry` 并列明所需入口。数据完整、计算通过后直接交付，无需因指标好坏请求追加实验。

## 6. 最终只交付一个小表

执行端内部数据留 `/home/scv/shared/phasemet_1_20261015/`。**只提交 `phasemet-1-result.md`，内容仅下面一张表。** 六项比较各一行，四个重点案例各一行，最后一行写运行状态。

|来源／对象|比较或案例|旧定义mIoU差 pp|参考类固定定义mIoU差 pp|差值符号改变组数／总组数|实际误分变化|备注|
|---|---|---|---|---|---|---|
|81|LM−LL||||||
|81|LM−MM||||||
|81|ML−MM||||||
|84|F真实−R4||||||
|84|F真实−F打乱||||||
|84|F真实−LL||||||
|81|T20WNV：LM−LL||||||
|84|T51TUM：F真实−R4||||||
|84|T50PLR：F真实−R4||||||
|84|T44UNA：F真实−R4||||||
|运行状态|完成／部分完成／阻塞|—|—|—|—|总耗时；原指标复现／计算检查；异常；执行端目录|

填写约定：

- 六项比较的mIoU差为32组等权平均，误分变化为全部共同有效像元的错误计数差及占比差；二者不能互相替代。
- 单案例行填该组差值，“符号改变组数／总组数”写0/1或1/1，误分列写该组计数变化。
- 备注简述参考1/2/3类分层中是否有差异，以及T32NKF是否仍有真实混淆损伤；这些明细内部计算，不再单独提交表格。
- 保留所有负差；缺值写NA及原因。若校验失败，填写已完成的行，状态行说明哪项失败。
- 原矩阵、完整逐组结果、排序、测试和复算记录留执行端，需要时再查；不提交README、审计JSON、manifest、图、ZIP或日志附件。

接收文件：`outputs/phasemet/phasemet_1_20261015/phasemet-1-result.md`。旧81/84文件、指标及历史判定不改，不要求论文解释。
