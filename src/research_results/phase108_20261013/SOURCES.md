# 一手来源与审读位置

全部访问于2026-10-13（北京时间）。下载成功的原始字节SHA256见source_manifest.json，文件留本地sources/；Git仅小型审读交付，不搬运论文全文。download成功不等于全文审读成功。

- P1：[arXiv v2](https://arxiv.org/abs/2411.13127v2)，PDF pp.3-7，III、IV-A/B/C，Table I和消融Table III-VI；[正式DOI](https://doi.org/10.1109/TGRS.2025.3597410)，Crossref title精确匹配；[作者代码](https://github.com/XavierJiezou/Cloud-Adapter)。论文与当前代码版本分列。
- P2：[NASA原文](https://ntrs.nasa.gov/api/citations/20210014061/downloads/Masek_Comparison_Cloud_Detection.pdf)，只取得正文索引片段§2.5/§4.2，后者印刷p12；[NASA正式记录](https://ntrs.nasa.gov/citations/20210014061)，[DOI](https://doi.org/10.1016/j.srs.2020.100010)。RF场景隔离未核实，不把点级精度提升当独立场景证明。
- P3：[出版商正文入口](https://doi.org/10.1016/j.rse.2024.114122)，索引正文§2.1.1/2.1.3（训练来源/隔离）、§2.4/2.5（模型）、Table A.4（作者成本）；[作者机构摘要](https://research-repository.uwa.edu.au/en/publications/clouds2mask-a-novel-deep-learning-approach-for-improved-cloud-and/)；[作者代码](https://github.com/DPIRD-DMA/CloudS2Mask)。processing_settings.py的required_bands是当前13输入的实现证据。
- P4：[公开正文v1](https://arxiv.org/html/2111.08349)，§III-A/B/D/E/F、IV、VI、Appendix A；[正式DOI](https://doi.org/10.1109/TGRS.2021.3128280)；[作者代码](https://github.com/aliFrancis/SEnSeI)，README Datasets直接列4172871及40:10:50。
- P5：[作者说明](https://shi-eo.github.io/publication/2025-fmask5)，摘要不是正式正文；[正式DOI](https://doi.org/10.1016/j.rse.2026.115672)，[出版商记录](https://api.elsevier.com/content/article/pii/S0034425726004426?httpAccept=text/xml)，只有coredata；[作者代码](https://github.com/GERSL/Fmask)，README Table1、Complete Package、Version History、Reference。正式DOI于2026-09-26登记，登记时间不是经核实的首次在线发表时间；正式刊期2026-12-15。
- P6：[NASA原文](https://ntrs.nasa.gov/api/citations/20220006382/downloads/rse-2022-112990.pdf)，本次从[作者机构DLR同篇PDF](https://elib.dlr.de/187708/1/1-s2.0-S0034425722001043-main.pdf)取得；PDF pp.4-5 §2.1.3-2.1.6、§2.2及Appendix A Table A1；[DOI](https://doi.org/10.1016/j.rse.2022.112990)。本项目点样本FPR不可外推全景是依据抽样支持的统计推断，不冒充原文结论。
- P7：[出版商摘要](https://www.sciencedirect.com/science/article/pii/S0034425725000987)，[DOI](https://doi.org/10.1016/j.rse.2025.114694)；[作者代码](https://github.com/DPIRD-DMA/OmniCloudMask)，README Quick Start、MODEL_CHANGELOG、model_download_links.csv。V1对应论文，当前HEAD已含V4；本次仅对代码读取到的条件作可行性判断，正式正文训练/评价隔离细节unknown。

检索以六篇题名/DOI、作者链接和直接后继为界，只纳入一个新增论文。未把搜索中出现的无关SEnSeI同名项目或其他方法纳入依据；不证明截至今日没有其他相似方法。
