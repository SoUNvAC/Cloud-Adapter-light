# Phase320：错Shadow标签由什么组成，CB的召回增益能否靠调阈值获得

## 1．做什么与交付范围

一次完成两个收尾检查，只读已有缓存与参考，**gzs CPU执行，零训练、零拟合、零神经前向、零下载**。

1. SPARCS、L8的Source Q90 Shadow伪标签中，错标实际对应什么参考类别？是否集中在参考边界附近？沿用Phase319几何状态后，各部分有什么差异？
2. CMC、SPARCS、L8中，普通PL只调整Shadow判断阈值，能否达到CB原始读出的召回／误报取舍？

本轮是已有开发数据上的事后诊断，不是新方法成绩、独立确认或盲测。两个检查一起交付，不再拆成准备phase。当前Flip和几何候选的历史结论保持不变；不修SPARCS投影、不改高度角度、不追加训练。是否提出下一种方法，待这两项结果一起讨论。

沿用方法窗口：2026-10-26T20:52:39.386137+08:00至2026-11-09T20:52:39.386137+08:00，不重新计时。该窗口仅针对本次方法探索，不是全项目自动停止日期。

## 2．先锁输入和分析定义

先写 `ANALYSIS_LOCK.json`，包括本计划SHA、代码SHA、输入文件SHA、成员清单、下述分组与阈值规则、已知317—319结果及预算。可以读取已有质量报告；不得把这次后续分析称为从未看过答案的预注册。

输入从既有manifest定位，先查实际路径，禁止猜路径后悄悄换缓存：

- `src/research_results/phase317_20261026/` 的METRICS、参考及评估manifest；远端原工作目录 `/home/scv/shared/phase317_20261026` 和已验收备份。
- `src/research_results/phase319_20261027/` 的 `NATIVE_STRATA.csv`、`PRECISION_COVERAGE.csv`、几何状态与选择掩膜、参考身份和坐标核验记录。复用已经生成的数据，不重新调几何。
- 实现参考 `src/tools/worker_phase317.py` 与 `src/tools/reference_phase317.py`。评估缓存通常只有 `pred,valid,shadow`，其中shadow是 **softmax(s)[2]**；fit教师另有完整scores与pseudo。必须逐文件核实，不把q_Shadow与shadow混用。
- Source身份沿用SHA `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`；本轮不加载模型运行。

输入缺失可查既有备份恢复，不能新增推理。关键身份不能核验则相应检查交付 `not_testable` 和缺项，另一项继续。

## 3．检查一：拆清错误Shadow伪标签

### 3.1 固定分析支持

- 只看Phase319同一批fit：SPARCS60图／60scene，L8固定100图／18scene。不改样本，不用val挑例子。
- A_Sh固定原Source Q90接受的Shadow：image_valid、sum(s)>1e-12、argmax(s)=2、q_Shadow>=0.90。完整计数应为SPARCS152450、L8426552。
- 错标E=A_Sh且参考有效且truth!=Shadow；正确T=A_Sh且参考有效且truth=Shadow。无效参考单列，不当错标。每池必须复现319原集合的有效数、正确数和precision，再做新分解。
- 不只统计几何准入子集：完整A_Sh必须保留，避免SPARCS缺失的33.69%被隐藏。几何状态沿319原定义：supported、unmatched_observed、unmatched_unknown、input_not_qualified；若旧记录不能区分，记status_unavailable，不重新计算投影补齐。

### 3.2 真值类别：只说标签实际支持的话

| 数据池 | 原始标签 | 本轮可报告的语义 | 父类 |
|---|---|---|---|
| SPARCS | 0、1 | shadow、shadow over water | Shadow |
| SPARCS | 2、3、4、5、6 | water、snow、land、cloud、flooded | Surface、Surface、Surface、Cloud、Surface |
| L8 | 64 | shadow | Shadow |
| L8 | 128、192、255 | clear、thin cloud、cloud | Surface、Cloud、Cloud |
| L8 | 0 | fill，无效参考 | ignore |

复用319的NATIVE_STRATA并交叉核对总数。L8 raw mask坐标沿319已经验收的转换；fill不能因旧PNG转换被当clear。SPARCS缩放沿原512最近邻，不改回原分辨率统计。

**不能从这些标签推导的项目：** L8 clear中的水体、暗地表、地形阴影；SPARCS land中的暗地表和地形阴影；标签是否画错。它们统一记“当前参考无法区分”。biome只是scene属性，不能代替像元类别。亮度低、靠近边界、看图像像水体，都不能升级为已证实的错误类型。本轮不另建人工真值、不引入光谱阈值分类。

### 3.3 位置：固定一个边界宽度

在各自原评估网格计算两张参考边界：Cloud／非Cloud与Shadow／非Shadow。对4邻域相邻、双方参考有效且类别二值状态不同的像元，将双方标为边界，再按Chebyshev距离膨胀 **2像元**。不扫描其他宽度。

距图像外沿或任一参考无效像元<=2像元的候选，单列 `boundary_context_unknown`，不硬分为远离边界。其余候选按两个边界标志分为四个互斥位置：neither、cloud_only、shadow_only、both。

交叉统计原始真值类别×这五种位置×319几何状态。对每格同时保留E、T、有效A_Sh数和precision；既看错误数量，也看正确Shadow会不会被一起删掉。不同数据池的2像元物理尺寸不同，不能混成统一米制距离。

“靠近参考边界”仅说明空间位置，不能改写成“标注偏差”或“薄云边缘”：薄云只由L8原始192标签支持。表格无法识别的原因保持未知。

### 3.4 要交的表

- `ERROR_COMPOSITION.csv`：pool, level(image/scene/pool), unit, native_class, parent_class, accepted, reference_valid, correct_Shadow, wrong_Shadow, invalid_reference, share_of_all_errors_pct, precision_pct。缺失类别保留0；分母为0的比例NA。
- `ERROR_LOCATION.csv`：同一层级与身份字段，另加native_class, boundary_bin, geometry_status, accepted, valid, E, T, share_of_all_errors_pct, precision_pct。各互斥格必须回加到原全集；geometry_status不可用也入表。
- `ERROR_SCENE_SUMMARY.csv`：每scene的E、T、water错误、thin_cloud错误、其余Cloud错误、可知Surface错误、无法细分Surface错误、近任一参考边界E、远离两边界E、边界上下文未知E、几何状态覆盖。注明哪些字段重叠，禁止相加当新总数。
- `ERROR_OPPORTUNITY.md`：用一页回答“已知的大块错标是什么、涉及多少scene、是否存在不用GT就能获得的区分信息、现在缺什么”。只作机会盘点；水体占比大本身不证明几何有用，不自动获准调几何或训练。

## 4．检查二：PL加Shadow阈值的对照

### 4.1 比较对象和缓存边界

固定Phase317的PL与CB、同一seed30201、原val支持：CMC32组、SPARCS10scene、L8963图／8scene。沿原参考ignore、图像有效性和scene归组。不得把L8963图当963个独立scene。

先从原 `pred` 复现PL/CB的macro Shadow recall、macro FPR、pooled recall/FPR、Shadow IoU，以及有效像元和组数；与METRICS.csv相符才继续。数值容差1e-6个百分点，整数混淆必须一致。支持不一致先报告并查明，不能偷偷取容易的交集。

本轮阈值定义为 **predict_Shadow = (shadow >= t)**，每池一个全局t，所有scene共享。shadow来自已存softmax(s)[2]，不用fit的q做评估阈值。

缓存通常没有完整三类scores，因此本检查是Shadow对非Shadow的二分类读出。不能从单个Shadow分数恢复精确的三类logit偏置，也不能给原Shadow改判时虚构Surface／Cloud归属。阈值读出的三类mIoU记NA；原三类argmax成绩作为已有基线单列。三类argmax点不一定等于t=0.5，不得声称它们相同。

### 4.2 固定两个问题，精确处理分数并列

按每池PL缓存的所有不同float32分数构造精确阈值阶梯，包括全不选和全选。相同分数必须整块纳入，不允许按GT或像元顺序拆并列。非有限值先报输入错误；不得用分数分箱、抽样或每scene最佳阈值替代主分析。

macro recall只平均有正例的原scene；macro FPR只平均有负例的原scene，与317完全一致。可对正像元赋权1/(有正例scene数×本scene正例数)，对负像元赋权1/(有负例scene数×本scene负例数)，按分数累计；分母从固定参考一次生成，不能随阈值改变。

主要对照目标取自 **CB原三类argmax读出**，并在ANALYSIS_LOCK写下数值：

1. **相同误报预算：** 在PL全局阈值点中，找macro FPR<=CB原macro FPR时最高的macro recall。多个点recall相同，选FPR更低者，再并列选较高t。报告它与CB的召回差及实际FPR差。
2. **达到相同召回：** 在PL点中，找macro recall>=CB原macro recall时最低macro FPR。多个点FPR相同，选recall较高者，再并列选较高t。报告它与CB的FPR差及实际召回差。

每项另交紧邻预算两侧的阶梯点与并列块大小，解释无法精确匹配时的离散差距。不得把线性插值点当作实际可部署阈值；零预测时precision=NA、recall=0（有正例），FPR依定义。Shadow IoU由二分类TP/(TP+FP+FN)计算。

对CB的连续shadow也按同一规则生成对应曲线摘要，附两臂原有macro／pooled AP复现，用于观察排序是否变化。无需导出所有阈值的海量CSV，只持久化规定比较点、相邻点、输入／程序SHA与精确算法说明。不得事后增加最优mIoU、逐组最优阈值、温度或偏置拟合。

这些阈值使用了val真值，必须命名 `oracle_diagnostic`。它回答“事后调阈值是否足以复现这项取舍”，不是可推广的新适应方法。若PL能在不更高误报下达到CB召回，可说CB默认点的二分类召回优势在本池可被阈值调整复现；不能说CB所有训练效果都等同阈值移动。若不能，也只说明这些缓存下单个全局阈值不够，不能直接推出云特有创新成立。

### 4.3 要交的表

- `ORIGINAL_POINT_REPRODUCTION.csv`：pool, arm, groups, positive_groups, negative_groups, support, TP,FP,FN,TN, macro_recall_pct,macro_FPR_pct,pooled_recall_pct,pooled_FPR_pct,Shadow_IoU_pct, 对317误差, pass。
- `THRESHOLD_MATCH.csv`：pool, arm, target(CB_original_FPR/CB_original_recall), point_kind(selected/lower_neighbor/upper_neighbor), threshold, selected_pixels,tie_block_pixels, TP,FP,FN,TN, macro及pooled recall/FPR, precision,Shadow_IoU, 与CB原点的recall/FPR差, budget_satisfied, interpretation。端点用明确字符串，不把Infinity当标准JSON数值。
- `THRESHOLD_SCENE.csv`：各主要选定点在每个原scene的支持、正负例、混淆、recall/FPR/precision/Shadow IoU；无正例或无负例指标NA，scene不能被删去。
- `RANKING_SUMMARY.csv`：pool,arm,macro_AP,pooled_AP,原AP差,exact_unique_score_count,原点及规定阈值点摘要。AP沿317现有定义，不能换积分口径制造差异。

## 5．预算、正确性与恢复

- 计划gzs CPU总时长<=2小时，另给1小时实现修复／复验余量；峰值内存<=16GiB。逐池顺序处理，必要时精确外部归并排序，保持并列块完整。GPU调用数必须为0。
- 新持久交付<=1GiB，不复制全部旧缓存；临时排序空间<=6GiB，启动前核验至少8GiB可用。超预算交部分结果与具体缺口，不改成近似算法充当精确主结果。只清理本轮明确登记的临时文件，保留失败日志与旧原件。
- 用小型合成数组验证：并列分数整块、全选／不选、仅正／仅负scene、宏平均与池化差异、两种预算选择；边界测试有效／无效邻居与图像外沿。不要为真实结果追加参数。
- E+T+invalid_reference=完整A_Sh；互斥类别／位置／几何格回加一致；scene回加池化整数一致；阈值降低时TP/FP单调不减；排序法与小数组暴力阈值枚举一致。
- 统计使用固定参考；不得回写教师缓存、旧报告、旧标签或模型。没有本轮输入资格的部分如实NA。

## 6．最终交付与下一步边界

目录 `src/research_results/phase320_20261027/`；若实际执行日期改变，目录用真实日期并在manifest说明，不改phase号。

交付本计划规定的全部表、ANALYSIS_LOCK、输入SHA清单、运行代码、检查记录、成本记录、失败／恢复日志、`phase320.md`执行记录和 `RESULT.md`。RESULT开头用通俗中文回答：

1. 错Shadow标签主要错成什么？其中多少是参考能直接证明的类别，多少原因仍未知？
2. 普通PL调一个阈值，能否在相同误报预算下找回CB那么多Shadow？分别报告三池，不能只挑成功池。
3. 是否出现值得再提出一次有限实验的新信息？指出具体输入证据、受影响scene和待验证假设；没有就建议关闭当前方法探索，整理阶段报告或换方向。

本轮不设“边界占比高就证明标注错”“水体多就几何go”的机械标准；不根据结果反复修改停止条件。结论仍遵守EXPLORATION_POLICY：区分负结果、未知与明确新线索。任何新方法尝试须另写固定方案、对照与上限，本轮不自动执行。

完整备份本轮交付成员并SHA验收；双机同步按既有执行流程记录真实提交和校验，不伪称只核读报告已复验全部原件。不设置定时任务，不自动启动Phase321。若后续写论文，保留317—320全部候选、失败、控制与开发使用历史；本轮本身不补SHOT/TENT训练，也不承诺投稿去向。
