# Phase201之后：本地证据与文献对照筛查

日期：2026-10-17。仅阅读历史报告／训练代码并检索文献；没有训练、推理、读取影像像元、重算指标或启动新phase。本文件是选题判断，不是执行协议。

## 当前判断

**目前没有一个可以直接宣布成立的新课题。仍值得追查的，是已有适配损伤的具体来源；优先查归一化和读出变化，而不是继续堆模块、改评价规则或替Omni寻找预定缺陷。**

Phase201没有观察到被支配的选中模型。A、C在指定预算目标下有条件召回损失，区间跨零且存在风险取舍；B是预算迁移失配；强基线D没有相应被支配选择。因此不再以“常用评价反复选错模型”作为主论点。

下面的优先级表示低成本厘清问题的顺序，不表示新颖性已经通过确认或承诺投稿结果。旧固定训练方案保持关闭。

## 1．最值得先厘清：encoder冻结后，适配损伤来自哪里？

### 本地已有事实

- Phase122—123两个C100在Catalogue的32TML／40XDR及PixBox的19FDV／54TWN重复出现AP损伤。54TWN仅7个正参考点，不能靠其大幅度单独支撑机制。
- `src/tools/train_phase122.py`：只更新decoder，encoder参数和buffer冻结；`student.train(); encoder.eval()`。代码明确记录decoder BN buffer可以变化，梯度验收后恢复初始decoder，再正式训练。
- 同一文件训练使用512裁剪，在每个裁剪上调用`mu.channel_norm(a,0)`；`src/tools/infer_phase122.py`使用作者1000／300整景滑窗。**训练／推理窗口不同是事实，造成损伤尚未证明。** 原模型本身的训练增强与窗口分布也须核验，不能只因存在尺寸差异就认定错误。
- Phase125归档确认总体取舍与重复损伤，但没有识别具体原因。Phase201也没有把这些模型判为全面更差。

### 文献已经研究了什么

1. Kumar等，ICLR2022，*Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*：微调可能损坏预训练特征，提出先线性探测后微调等思路。[原文入口](https://arxiv.org/abs/2202.10054)
2. Wortsman等，CVPR2022，*Robust Fine-Tuning of Zero-Shot Models*：研究微调与域外鲁棒性取舍，提出WiSE-FT权重组合。[正式论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf)
3. Kethavath与Hu，2026，*Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift*：CLIPSeg／CloudSEN12+中，少标签微调会在光谱含混类别上暂时下降，整体mIoU可能掩盖该现象。arXiv注明EarthVision @ CVPR2026，不能称CVPR主会。[作者预印本](https://arxiv.org/abs/2604.08956)

**不能再以“首次发现少样本适配伤害云影”立题。** CLIPSeg弱零样本起点与Omni专业预训练起点不同；C100冻结encoder，不能未经证据套用“encoder特征被破坏”的解释。encoder不变也不等于整网表示不变，decoder仍有可学习表示与统计。

### 具体值得问的问题

在当前冻结encoder设置中，重复损伤主要对应：

- decoder学习到的判别规则变化；
- decoder BatchNorm运行统计变化；
- 训练裁剪／推理窗口的输入归一化统计与内容组成变化；
- 或参考定义、场景分布和稀疏支持造成的表观差异？

这些解释可能共存。不能把跨方法同tile损伤当共同机制，必须先核对同一产品、参考与支持。要同时保留改善组和稳定组，不能只研究最差例。

**下一项合理工作是全文核读和现有模型状态清单核验，不是重训C100。** 只在某个机制有本地证据且能与已有工作明确区分后，才讨论固定的小规模机制对照；新归一化变体也不能自动作为新主方法。

定位：本地证据联系最强，值得优先读文献；是否有论文增量仍未知。

## 2．次优先：Omni的patch归一化是否带来有意义的输入范围依赖？

更精确的问题是：**固定模型、分辨率和目标局部内容后，仅改变归一化统计的取样范围／周边组成，会不会改变本应一致的有效观测判定？**

Omni官方明确按patch逐通道计算均值／标准差，并将其作为跨传感器泛化的优势；作者也已研究小patch上下文不足。[方法说明](https://omnicloudmask.readthedocs.io/en/latest/how-it-works.html)｜[上下文基准](https://omnicloudmask.readthedocs.io/en/latest/spatial-context.html)

直接相关文献：Buglakova等，ICCV2025，*Tiling artifacts and trade-offs of feature normalization in the segmentation of large biological images*，已经研究归一化引起的分块伪影和train／eval差异，并提出检测指标及BatchRenorm方案。[正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Buglakova_Tiling_artifacts_and_trade-offs_of_feature_normalization_in_the_segmentation_ICCV_2025_paper.pdf)｜[作者教程](https://buglakova.github.io/tiling_artifacts_tutorial/)

区别必须核实：输入patch标准化与网络内部feature normalization不是同一操作，但都可能造成范围依赖；前者并非因此自动成为无人研究的新问题。

本地Phase118及SCD只比较若干分辨率／窗口设置，不能分离归一化、上下文、有效分辨率及拼接影响，也没有证明Omni存在严重不稳定。Phase200固定点V1／V4类别一致，不提供此类失效的正证据。

若未来进入机制核验，应分别控制输入归一化与网络内容／拼接，使用真实相邻影像组成，并把分数变化与真实错误、有效像元损失分开。人工拼接一个极端外围只说明可扰动，不说明正常部署有问题。尺寸扩大不能保证排除上下文影响，也不能仅凭“离中心远”宣称因果隔离。

定位：问题较清楚、可能不需训练，但尚无本地失效证据，且与ICCV工作有明显重合。**只建议先读这篇，不重启Phase118窗口搜索。** 若最终只是已知分块伪影的本地复现，归入工程说明。

## 3．有实际价值但暂不优先：云影残留如何影响可用时序观测？

从任务意义看，云影是否造成某个可复核的变化检测误报、物候异常或水质偏差，比单独提高AP更具体。但本地现有数据主要是单景参考、少量点和模型分数，缺少匹配的时序／下游真值；不能把AP或平滑度直接改名为实际效用。

已有研究明显接近：

- *A globally applicable deep learning model for Sentinel-2 cloud and shadow detection*，Science of Remote Sensing12(2025)100278，使用78个全球tile的一年11,458景检查保留地表反射率的时序一致性；摘要同时报告最高整体精度与并非最高云影F1。[正式入口](https://doi.org/10.1016/j.srs.2025.100278)
- *Assessing the Impact of Threshold on Cloud Masking Algorithms for Downstream Tasks*，ICLR2026 ML4RS Workshop，用对象数量／熵等代理终点评价阈值。[论文](https://openreview.net/pdf?id=0X7yZncUk6)
- 本地已全文复核的Ducar水体案例证明有应用动机，但Phase200没有在锁定清晰岸边点复现其误报，不沿原案例继续挑点找错。

若以后走此路，必须先明确一个可获得真值／独立参考的应用对象，并同时约束丢弃有效观测的代价；越平滑不等于越正确，真实扰动也可能造成反射率突变。它需要新数据和领域合作，资源投入明显大于历史归因，不能包装成现有材料立即可完成的课题。

定位：应用意义有支撑，原创问题和数据可行性未确定，暂列储备。

## 4．本轮发现的新近重合：校准迁移也不是空白

Tsai与Chung，*Conformal Risk Control under Sentinel-2 Product Shift*，SSRN预印本，2026-09-18发布。[作者投稿入口](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7467639)

摘要报告CloudSEN12配对L1C／L2A、相关场景分组、多seed和校准划分，发现冻结校准规则迁移后风险上升，并用产品相关标准化／分位数迁移分析预处理来源；冻结DOFA对照的表现又不同。

这里的joint risk是“接纳且错误”的概率，**不是本项目非云影条件FPR**；其主要模型是逐像元光谱模型，不能直接等同Omni。文献为预印本，不按正式同行评审证据对待。其模型差异也提醒：不能从G7的尾部误报推论所有强网络都有同样问题。

这项工作已直接覆盖“遥感产品变化—归一化—冻结校准风险”链条。建议阅读全文确认，暂不立“把conformal套到云影阈值”项目，也不把我们的旧经验校准失败当作其有限样本理论被推翻。

## 5．本轮资源排序

| 候选 | 本地现象 | 文献重合 | 当前资源判断 |
|---|---|---|---|
| 冻结encoder下decoder适配的重复损伤来源 | 有重复损伤和代码事实，未归因 | 微调损伤已知；云类别少样本退化也已有 | **优先做全文／状态清单核验**，不训练 |
| 输入归一化范围与内容依赖 | 仅有方法依据，无已验证失效 | ICCV分块／归一化及Omni上下文已有 | **次优先读全文**，不预定Omni有错 |
| 云影残留造成特定时序／应用损失 | 本地下游证据不足 | 时序与下游评价已有 | 有合作／真值再考虑 |
| 平均／尾部风险和conformal修正 | 旧方案风险存在，强基线尚未建立问题 | 已有直接遥感预印本及一般风险理论 | 暂不追加方法 |
| 跨数据集标签语义差异 | ALCD疑点和修订差异存在，未独立裁决 | CloudSEN12等早已讨论标签协议与类别定义 | 作为必要排除项，不另立泛泛benchmark问题 |
| GPU计算与整景速度脱节 | Phase119有定量证据，具体瓶颈未定 | 官方已有预取／batch／拼接优化 | 工程／学位论文可用，先不当TGRS主问题 |
| 选模评价规则普遍失效 | Phase201不支持强论点 | 重合充分 | 收口；保留评价章节 |

## 6．下一步文献优先队列

优先核读三份，不要求先下载更多：

1. [Buglakova，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Buglakova_Tiling_artifacts_and_trade-offs_of_feature_normalization_in_the_segmentation_ICCV_2025_paper.pdf)：确认输入／feature归一化、运行统计、train／eval差异和已有因果对照覆盖范围。
2. [Kethavath与Hu，2026](https://arxiv.org/pdf/2604.08956)：确认类别退化发生条件、seed／标签采样、LoRA对照及是否解释了损伤；不能只据摘要认定我们有不同机制。
3. [Tsai与Chung，SSRN2026](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7467639)：核对产品、模型、风险定义、预处理与校准的分离方式。

Kumar2022和WiSE-FT作为成熟背景；全球Swin-Unet时序论文在准备应用方向时再优先读。先把上述三篇与本地decoder冻结／BN／窗口链对照清楚，再决定是否存在可检验的新增问题。若没有，接受文献已经解释大部分现象，不再靠换标题延长试错。

## 阅读范围与本地依据

本轮新文献核验以作者／出版方摘要、正式PDF可检索片段和官方方法文档为主，**未宣称新找到论文已全文精读**，不能据此完成系统综述或证明无人做过。ICCV PDF核对了摘要与输入／feature归一化分类的正文片段；其全部对照仍待核读。部分出版方全文接口受限。

本地阅读：

- [三类结算](../RESEARCH_THREE_WAY_SETTLEMENT_20261017.md)、[历史日志](../EXPERIMENT_LOG.md)。
- [Omni三篇既有全文复核](OMNICLOUDMASK_THREE_PAPERS_FULLTEXT_REVIEW_20261017.md)。
- [Phase125总结](../../outputs/phase125/phase125_20261017/BRANCH_SUMMARY.md)、Phase124逐组证据。
- [Phase201结果](../research_results/phase201_20261017/README.md)、逐条选模回放CSV。
- [训练实现](../tools/train_phase122.py)、[推理实现](../tools/infer_phase122.py)，本轮只阅读代码，没有执行。
- [SCD结果](../../outputs/phasescd/phasescd_1_20261015/phasescd-1-result.md)。Phase200候选点结果按用户已交付陈述引用，不声称本轮新核验其全部封存件。

本文件不修改三类结算的历史事实，不将原来“最高优先”的评价种子继续当作已成立主方向，也不恢复已关闭的固定配方。
