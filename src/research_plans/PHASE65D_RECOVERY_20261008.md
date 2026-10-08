# 65D小型恢复实验：fit拟合、development评价

用户明确授权固定比较决策/校准、MsRE+Source、再增加NIR/SWIR三个层次。只拟合低容量像元分数组合，不重训分割网络、不换checkpoint、不读任何65b/T18/原65a确认/65c/旧sealed像元或结果。沿用原153 fit相关组代表、32 development代表（含Shadow仅7组），ALCD不参与。development已多轮查看且原2000步checkpoint由development mIoU选定，本轮是事后恢复探索，不是独立确认。

## 冻结设计

- Source SHA=64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9；MsRE最佳2000步SHA=82f88d235948ef0a9202cebaa026e7f9fb1e1ad5c0aa361d9045c3e6991ee16d。不使用3500/4000替换。原split SHA=a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b不变，原正式支持门不恢复。
- 级别1：logit(MsRE softmax Shadow分数)；级别2：再加logit(Source Shadow分数)；级别3：再加原生TOA B08/B11/B12。分数clip[1e-6,1-1e-6]再logit。三层均为带截距的一层逻辑回归，平均组加权BCE+0.001/2乘系数平方；截距无惩罚，MsRE和Source系数约束非负，光谱系数可正可负。不加交互、门控、非线性、场景ID、标签比例、RF置信图或人工质量特征，不选择超参/波段/权重组合。
- 每个fit代表从全部有效参考像元均匀无放回抽取最多8192个（不按类别平衡/选择困难样本）；每组总拟合权重相同。随机种子65081+产品SHA派生，三层共用同一抽样。均值/标准差仅fit样本按组权重计算，零方差缩放为1。固定SciPy L-BFGS-B maxiter500、gtol1e-7、ftol1e-12，未收敛即停止，不按development调试参数。
- 校准使用全部153 fit代表的有效非Shadow像元，固定全局阈值，使fit逐组平均FPR<=1%，严格保守处理ties。每层有独立fit阈值，但同一预算、同一组加权规则。预测>=阈值为Shadow，否则采用MsRE在Surface/Cloud间的原胜者；原Source/MsRE argmax作为未拟合锚点。1%仅是预定操作预算，不保证development也为1%；必须报告实际迁移FPR。
- 官方13波段cube中提取B08/B11/B12，与准备好的六波段image-valid及唯一标签有效mask共用格网；核验官方member SHA和准备数组SHA。保留TOA原尺度、不依据参考标签调变换。仅对153 fit代表与32 development读取。
- fit先提取两固定模型分数且逐景argmax confusion必须与既有fit评估一致；development复用已有65D分数缓存并核验SHA。先完成拟合/fit校准并独占写recovery_lock.json后才能装载development分数/标签作评价。锁绑定协议、代码、模型、manifest、fit样本和索引SHA；不覆盖旧阈值锁或结果。

## 固定评价和解释

- 主描述为7个含Shadow相关组宏AP及逐组配对AP增量；另外报告32组池化AP、oracle固定FPR1%/5%召回、冻结fit阈值实际逐组/池化召回与FPR、mIoU/Shadow IoU、预测Shadow比例。AP/固定FPR诊断使用全有效像元及精确ties；后者阈值参考标签依赖，不能当部署规则。25个无Shadow组仍完整参与误报。
- 固定比较2减1、3减2；报告所有7组原始差值、2000次组bootstrap描述区间（seed65310），不报告像元iid显著性。抽样只降低拟合成本，评价全像元；约一亿像元不变成独立样本。bootstrap条件于已拟合模型，不包含fit不确定性或先前development模型选择，不作正式确认。
- Source以上为完全独立锚点，不能因总体平均改善声称所有损伤恢复；可注明已知受损development场景，但不据此选择阈值或重拟合。评价后不修改本轮冻结策略，不读T18。
- 产物只在shared/data/sentinel2_cloud_mask_catalogue_4172871/phase65d_recovery_20261008，新shared报告/图交付本地outputs/phase65/phase65d_recovery_20261008；work_dirs仅只读复用，不自动同步/清理。原正式训练禁令和2026-10-20北京时间预算仍有效；本轮授权仅上述低容量恢复实验。
