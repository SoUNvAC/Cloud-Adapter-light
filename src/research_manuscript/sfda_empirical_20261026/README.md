# VFM云分割SFDA实证稿骨架

日期：2026-10-26。状态：依据Phase300—314已有结果的内部初稿；不是投稿完成稿，不增加实验。

更新：已纳入Phase315的三次初始化重复和Phase316的数据准入结论。连续正文已补齐引言、相关工作、实验、结果、讨论、结论和参考文献；完整阅读版已将原四张核心表及重复结果放入正文。

## 交付

- [中文导读](D:/Cloud-Adapter-light/src/research_manuscript/sfda_empirical_20261026/READING_GUIDE_ZH.md)：用简单的话解释结果、代价和当前限制。
- [完整阅读版](D:/Cloud-Adapter-light/src/research_manuscript/sfda_empirical_20261026/MANUSCRIPT_FULL_DRAFT.md)：连续英文正文、内嵌表格及十条参考文献。

- [贡献论证与补验决策](D:/Cloud-Adapter-light/src/research_manuscript/sfda_empirical_20261026/CLAIMS_AND_DECISION.md)：核心主张、证据边界、区别及最小确认顺序。
- [四张核心表](D:/Cloud-Adapter-light/src/research_manuscript/sfda_empirical_20261026/CORE_TABLES.md)：投稿式英文表题/表注，保留原始来源与分母。
- [英文正文源稿](D:/Cloud-Adapter-light/src/research_manuscript/sfda_empirical_20261026/MANUSCRIPT_DRAFT.md)：连续正文源文件，表格单独维护；运行assemble_full_draft.py可重新生成完整阅读版。
- `EDITORIAL_QA.json`：原CSV数值、关键算术、计数恒等及文件链接检查；不代表重算全部模型指标或重验远端SHA。

## 骨架完成后的判断

现有结果能组织成一个清楚的问题：**在此云/云影三类任务上，适配纠正与损伤分属不同类别转移，修复后者的约束可能大量撤销前者；只看一个总体指标或最大失败组会遗漏这些取舍。** Phase315已在固定日程的三个LoRA初始化上重复PL总体收益及两类转移方向；保护约束仍仅单seed，整体仍为单配置、历史开发池观察，不能据此认定TGRS贡献已充分。

PL的初始化重复已完成，不再重复立项。Phase316后，独立确认池准入仍为0，KappaSet本轮资料检查收口。下一缺口是可合法、可追溯地确认同一三类问题的数据支持；没有合格池不新增评价或训练，也不再自动重复相同资料核查。若未来独立确认不再出现类似类别取舍，原观察只能保留为当前配置个案；停止把它扩张成普适机制。协议、数据准入和资源需另立，不因本初稿自动启动。

## 编辑口径

正文按研究问题组织，不按Phase编号叙事。Phase编号只用于内部溯源。原四张表保留30201单seed端点，表1b新增PL三个初始化及样本标准差，不将SP/CM伪装为重复实验，不加显著性星号。缺类规则、宏/池化分母、训练有效更新、目标开发暴露和Source预训练交集限制必须保留。

本编辑任务未生成图、PDF、LaTeX或投稿材料，未新增训练、拟合、模型前向、定时任务或推送；Phase315由执行者完成，结果引用其封存报告。引言与相关工作已有可读初稿，但更充分的新颖性比较、正式强基线、干预重复与独立池确认仍缺，不能直接投稿。
