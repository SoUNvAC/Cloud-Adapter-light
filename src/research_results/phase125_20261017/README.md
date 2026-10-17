# Phase125：RegNet分支已收口

Phase119—125本轮研究总结与小型文档归档完成。没有新增训练、拟合、前向、下载、像元读取、阈值搜索或区间重算，也未补seed或选胜出模型。

数据结果与解释在BRANCH_SUMMARY.md分开：C100两seed平均宏AP仍低于R；Catalogue保留较低误报的工作点取舍，PixBox召回方向有seed波动。R/E的已测GPU模型耗时比36.4%，整景分段比93.2%，不能互相替代；C100匹配部署计时未测。

- BRANCH_SUMMARY.md：可供论文/学位论文改写的研究总结。
- RESULT_TABLES.md：完整逐seed、两工作点、历史背景及成本表，保留原精度。
- EVIDENCE_INDEX.md：输入、代码、模型、阈值与结果的SHA和位置；权重不复制上传。
- CLAIMS_AND_LIMITATIONS.md：可支持表述、不能支持的结论和未测项。
- CLOSEOUT.json：完成时间、范围、文件清单及零新增实验声明。

既有逐组主体表仍保留在Phase124，不删除、不重新打包历史大文件。不自动追加实验；新方向从126起，本分支收口不干预其他并行方向。
