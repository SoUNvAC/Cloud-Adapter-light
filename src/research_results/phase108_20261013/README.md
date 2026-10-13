# Phase108：文献定位与最小对照缺口

## 1 数据结果

完成六个指定入口与一个直接后继（OmniCloudMask）的限定审读/实现核验。阅读层级显式区分正文、正文片段、摘要与当前代码；无法核实的正式正文细节、权重大小、训练产品交集均保留unknown。文献及代码原字节本地留存SHA。**新训练/拟合/前向/校准/评价/指标重算/权重下载/依赖安装均0。**

交付：literature_matrix.csv、artifact_feasibility.csv、LITERATURE_REVIEW.md、STRONG_CONTROLS.md、NEXT_QUESTION.md、PROJECT_EVIDENCE.md、SOURCES.md及SHA清单。原始公开文档在本地sources/，仅小交付镜像至src/research_results/phase108_20261013，不同步work_dirs。

## 2 结论分析

当前最稳妥定位是冻结预测与浅层读出的受限实证应用。输出stacking、多光谱表示及云影深度网络已有先例；Phase107不支持强光谱特异性主张，原失败分枝和误报尾部问题保留。

最多选择OmniCloudMask与Fmask5-UPL作强对照。前者输入子集可满足，概率接口/权重版本/重叠尚需验收；后者完整输入与资源未齐。CloudS2Mask用过Catalogue/ALCD训练，且当前实现需13波段，不能当六输入新独立泛化对照。

唯一下一问题及成本/止损见NEXT_QUESTION.md。当前没有已核验的新未暴露评价池，也没有可据采样点证明全景风险的新增密集参考，因此本轮不启动下一实验。不承诺机制、新算法首次性或期刊录用。

限制：Tarrio仅正文片段；CloudS2Mask正文经出版商索引访问；Fmask5/Omni未取得正式全文。已交付定位与缺口，不称完整系统综述或已复现强基线。Fmask5正式条目已登记，刊期2026-12-15；在线发表日期未核实。
