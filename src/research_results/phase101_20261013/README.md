# Phase101：主结论与证据缺口收口

本轮只摘录既有最终报告：42行证据、17个阶段来源。零训练、拟合、校准、网络前向、新数据评价、区间重算及文献检索。数据结果见evidence_matrix.csv，结论讨论单独见CLAIMS_AND_GAPS.md。

- evidence_matrix.csv：E01—E42是数据行号，不含表头。每行一个比较，保留模型版本、fit／calibration来源、评价支持、历史判定与限制。
- extracted_values.json：精确JSON摘录、字段路径及完整来源报告SHA；CSV同时保留可读摘要和精确原始摘录。
- source_files_sha256.json及sources/：既有本地小报告、逐组表、冻结锁和验收回执的逐字节审计副本，不是新评价或work_dirs同步。
- VERIFICATION.json：本轮来源副本SHA、JSON→CSV字面值及支持边界核验。历史CPU回执仅收录，不重新执行历史验收或bootstrap。

## 数据阅读边界

差值字段名带pp时为百分点；原Phase67 ap.mean及其区间、概率率均值仍为0—1，须乘100才成为百分数／百分点。miou_percent已是百分数，勿再乘100。摘要不产生任何新差值或区间。

E02为Phase66原冻结候选的预设独立确认。E06—E07为Phase72失败的独立确认；不能以辅助指标翻案。其余development或已暴露ALCD池结果为探索性／事后诊断，区间不赋予新确认地位。原H1支持门失败保持不变。

E09／E13／E17等分别对应ALCD原版9组、原版共同6组、修订版共同6组；共同6组是同批场景，不增加独立样本数。E01—E33为153 fit模型，E34以后为100 fit／53 calibration模型，禁止跨版本归因或跨阶段平均。dev的32组FPR支持与7组AP支持分开。E36合成60m不能等同原生ALCD60m。E21、E40为oracle，不是冻结部署收益。

Phase89使用最终recovered_results；Phase90采用修复后交付；Phase94采用授权补拟合后的最终双模型；Phase99采用验收恢复后最终报告。原始报告与冻结锁不改。Phase85有独立数值验收回执，但当前根目录交付SHA清单未直接绑定报告文件；该报告SHA历史绑定标为“未核实”。较早阶段没有单独CPU回执的也明确标注，交付SHA匹配不冒充新CPU验收。

ALCD参考不是完整人工认证真值，主审身份／经验未核实、人工B未执行，54块缓存无故障前SHA的谱系限制保留。缺失身份、回执或支持不补跑。没有访问模型、概率缓存、影像或sealed test。

## 复现本次文字摘录核验

从仓库根目录运行 `python src/tools/build_phase101_evidence.py`，只读sources和提取规格。`--collect`只用于本次从已有本地outputs采集审计副本；远端不采集、不传回work_dirs。

唯一后续问题及继续／停止条件仅为提案，见CLAIMS_AND_GAPS.md；本轮没有授权新数据评价、拟合或文献检索。
