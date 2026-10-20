# Phase300 表注与定义

- 单训练seed30001，初始化seed20801；历史development/PixBox已用于开发；V4完整训练覆盖未知。结果不是新盲测或严格源域保持证据。Omni机制预实验不替代VFM骨干验证。
- 所有原相关组保留：Catalogue32；PixBox16产品/10相关组。Catalogue组均三类mIoU对相关组等权；池化值先相加混淆矩阵再计算。无影组仍进入mIoU与负类误报比例。
- 混淆矩阵：行真值、列预测；0=Surface，1=Cloud，2=Shadow。每类IoU的union=0时为NA；mIoU在union非零类上取均值。AP要求至少一正一负，否则NA；召回无正类为NA；负类误报比例无负类为NA；宏指标对定义有效组取均值，并报告有效组数。精确度无Shadow预测为NA。
- 绝对AP、IoU和率以%报告；差值为百分点。训练覆盖/探针以fraction报告。PixBox只计算参考点指标和负参考点的Shadow误判比例，不计算全景mIoU、IoU或FPR；相应mIoU/IoU列填NA。
- 默认决策为四类raw argmax，再按[Surface,Thick,Thin,Shadow]→[0,1,1,2]合并Cloud。AP分数沿用官方Shadow口径：clip(softmax(raw)+0.001,0.001,0.999)，并用官方输入有效掩膜。没有53组阈值校准，也未按学生评价结果改阈值。
- 训练伪标签使用完整1000/300教师raw-logit拼接，稳定softmax后相加Thick/Thin概率；三父类最大概率>=0.90且三输入有效才接受。训练概率不加0.001、不clip。父类Cloud logit为logsumexp(Thick,Thin)。CE只对接受像元求均值；全ignore不前向、不反传、不optimizer.step，也不补步。
- 纯影像支持仅B04/B03/B8A均有限、非零、非float32(65535/10000)。历史监督/评价支持包含标签ignore和更多波段规则；FIT_SUPPORT_DIFFERENCES.csv在双末步封存后的本地审计才连接历史支持计数。
- 32次探针=4产品×4时点×2臂。预测占比不是正确率。相对教师变化率在共同图像有效支持计算；教师来自完整上下文，学生来自中心512裁剪。最终真值审计另交三类分解，使用image-valid与唯一真值交集，不回流训练。
- Catalogue95%区间：按32相关组配对bootstrap10000次、seed300020；每次重算组均及池化混淆指标，不把像元视为独立样本。区间仅描述固定单seed模型；不增加辅助指标的显著性声明。PixBox mIoU及其区间为NA。
- 表五原始地表字段按原始definitions.txt全代码锁定，含零支持及缺失类别。SURFACE_TYPE_ID是辅助数据字段，其他列按原始说明记录。Bright turbid water与Dark water直接来自WATER BODY CHARACTERISTICS_ID；雪/盐湖/沙地名称不代表实测亮度。不同字段的分层重叠，不能相加。无模型分数推断地表类别。
- 两池均Sentinel-2，本轮没有跨传感器验证。PixBox B04/B03从10m按历史bilinear uint16规则到20m，B8A为20m；未做B08替换试验。Catalogue原subscene更早的原始重采样细节本轮未重建。
- 标签：Catalogue固定三通道Surface/Cloud/Shadow，未恢复原Thin/Thick细类。PixBox沿用固定C/S/U映射；S2地形/树/建筑影为地表负类，S3地表云影为正类，S4云上阴影等冲突进入ignore。两个标签源并不保证地形阴影或边界定义完全等价，池间性能差异不能单独证明标注偏移。
- 表六为CPU oracle：每池按学生默认全组等权负类误报比例设预算，在R4连续分数上取最低可行全局阈值；并列分数整批处理，无逐景阈值或插值。召回仅平均有正参考组。记录实际预算差；阈值来自评价标签，不可作为部署结果，不回流训练或checkpoint选择。
- 训练/教师Python文件打开审计没有成功读取标签；该审计不覆盖原生syscall。操作代理在末步封存前人工预览少量本地PixBox参考记录的偏差另见phase300.md；不能声称整个操作过程标签不可见。
