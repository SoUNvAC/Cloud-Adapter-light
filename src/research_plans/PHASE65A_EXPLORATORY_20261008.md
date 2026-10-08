# Phase65a探索性试跑：2026-10-08用户明确授权

用户在暂停人工复核后确认“对，先做phase65a”，授权范围为先前说明的Catalogue固定fit/development、source与MsRE对照。此新协议只授权探索性任务，不恢复原主检验，不修改原失败split，也不授权65b/65c。

- 原split SHA256：a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b；只筛出既有153 fit组（155产品）和32 development代表，不重抽。185个代表用于离线诊断，开发指标受到checkpoint选择影响，不冒充独立确认。
- source SHA256：64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9。MsRE seed65、4000步、lr1e-4、batch1、FP32；稀疏注入2/5/8/11、token16、scale1e-3及既有head delta，不增加新机制。每500步在development选择最高mIoU，fit随机512裁块/翻转/既有颜色增广。
- 官方README明确三通道为Clear/Cloud/Cloud_Shadow，RGB为3/2/1；读取原13波段TOA float32，将RGB线性乘255匹配source预处理数值尺度，不截断大于1的反射率，不使用目录缩略图或置信图。RGB输入模型前按BGR约定交给既有ImageNet均值/方差预处理。
- 只读取fit/development产品数组；mask必须bool三通道，一热恰好一类且六个所需波段有限/非零时为有效监督，其他255忽略。记录每产品有效像元预算、成员SHA和准备后SHA；无有效监督即停。输出概率/光谱部署特征只用影像有效mask，不借GT有效mask或GT比例。
- 先执行数据准备及一个fit代表source前向/权重加载/可训练参数检查，随后source代表评估、一次MsRE训练、最佳development checkpoint成对代表评估和描述性汇总。H1报告有支持代表的描述性Pearson与明确分母；H2沿用12固定可观测量及固定logistic，类别支持不足不声称可识别。此探索不作正式确认bootstrap通过判定，也不触发后续阶段。
- ALCD不参与本次训练或确认，人工语义复核暂停。65a_confirmation、65b_confirmation、65c_final和旧sealed test均不读像素、预测或指标。
- 新work_dir为src/work_dirs/phase65a_explore_20261008，prepared数据只写shared Catalogue子目录；拒绝覆盖既有任务/数据。只有一个训练作业；启动前检查GPU及任务冲突。全部work_dirs自动备份/传回/同步取消仍有效，允许读取远端任务状态，产物由用户手动操作。
- 预算截止2026-10-20北京时间，当天不启动新训练，到期停止仍在运行的新研究训练任务并保留日志/状态，不删除结果。原预注册主支持门失败仍为原方案真实结论，探索性结果不能替代它。
