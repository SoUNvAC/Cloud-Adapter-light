# 65D后续：轨迹、损伤结构及额外信息筛查

用户明确授权三项诊断；不训练新分割模型、不改变2000步模型选择、不改原split/65D校准锁，不进原65a确认/65c/旧sealed。产物仅shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_followup_20261008，新报告及图交付本地，不同步work_dirs。

1. 固定复用Source与2000步缓存，新推理仅既有3500/4000步：SHA分别db13c9c2278804c73ac45082e0a6d5bccd6a9ff3748082b6be4d89898812983a、57ccd89a722363cb9e2213524c42a689fbf6cac0917fcaa66720779586003187。32 development全量Shadow AP、固定FPR0.1/0.5/1/2/5/10%召回、原argmax云影比例/mIoU、逐景宏指标。只保留这三个适配checkpoint，不能填补500/1000/1500/2500/3000步AP。读取完整训练日志的mIoU，只判断指标目标差异，不按T18换checkpoint。development轨迹先落盘冻结，再只对T18FYG追踪3500/4000，绑定development报告SHA，T18不参与任何选择。
2. 统计单位是原唯一相关组代表。development有Shadow的7组才有AP，其他25组只进入误报/预测比例。排序损伤标签固定ΔAP<=-0.01，改善>=0.01，其余稳定；三个checkpoint同组重复不算独立重复实验。记录持续损伤集合。用7组LOO固定ridge回归λ10预测2000步ΔAP，RGB影像均值/暗比例及Source输出九特征为基础，增加B08/B11/B12均值为固定补充。每折独立标准化、无标签比例输入、无超参数选择；报告逐组OOF与MAE对照折内均值基线。7组不能支持可靠部署分类器或正式增量显著性，不宣称可重复跨seed。
3. 若出现development排序下降，则对7个含Shadow组离线比较Source/MsRE分数与固定负RGB亮度、负B08/B11/B12、负三光谱均值的Shadow-vs-Surface排序，另固定Surface中亮度最低10%子集（ties纳入）。使用同官方13波段cube、相同六波段有效mask，全部像元，验证member SHA。无需拟合像元分类器，不选光谱组合/阈值，不把standalone光谱信号称为模型恢复。几何/大上下文无已核验输入，不伪造测试；仅有筛查信号时提出后续匹配信息探针，证据不足不设计新机制。

图先数据剖析再轨迹/逐景原始点展示及视觉/灰度检查；所有结果相对现有参考标签定义、单seed、development事后探索，不能称新独立确认。预算截止仍2026-10-20北京时间。
