# Phase319：太阳方向筛查

结论：**not_testable**。固定160图均已列入元数据表；两池的实际几何输入覆盖、精度和保留量如下。此为历史已暴露数据上的固定规则探索，未训练、拟合或新增神经前向。

| 池 | 几何输入覆盖A_Sh% | 选中/原A_Sh | precision% | 相对原集pp | 保留% | 同数量高分% | 反方向% | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
|SPARCS|66.3149|90929/152450|80.8290|3.0413|59.6451|79.3322|73.9163|not_testable|
|L8|98.4068|402425/426552|82.0364|0.2742|94.3437|82.3281|81.5832|no_go|

太阳角来自实际同scene MTL。原TIFF的CRS/affine与1000→512（SPARCS）或原512裁剪（L8）组合；地理正反投影和九点方向/尺度误差有独立核验。L8仅用原轴/原编码恒等核对100图，rawfill0从精度分母排除，仍计完整影像保留率分母。

高度200–12000m、角容差±5°、半像元对角线径向容差均事前固定。从候选Shadow朝太阳找Source预测Cloud；反方向为固定安慰剂。G_conf每图同k，原q_Shadow降序、row-major破并列。所有选择及核先封存，独立CPU进程随后读取参考；FFT计数经整数误差与固定无标签直接偏移抽验。

SPARCS：未过条件 input_90pct, precision_90pct, gain_original_5pp, scene_median_positive, scene_majority_positive, beat_conf_2pp；>=100有效选中像元的scene 23；可比较scene 26，中位增量 0.0000pp，正/负/零 7/12/7。反方向选中 51538，保留 33.8065%。参考覆盖见GO_NO_GO原始字段，零支持记NA。
L8：未过条件 precision_90pct, gain_original_5pp, beat_conf_2pp, beat_reverse_2pp；>=100有效选中像元的scene 14；可比较scene 14，中位增量 0.1014pp，正/负/零 9/2/3。反方向选中 308133，保留 72.2381%。参考覆盖见GO_NO_GO原始字段，零支持记NA。

完整逐图/scene/池计数、参考排除及coverage见PRECISION_COVERAGE.csv；原native子类见NATIVE_STRATA.csv（biome不是像元类别）；上下文完整性、supported但外部未知、unmatched_observed和unmatched_unknown分别保留。后者不能作负标签，Source漏云同样可能造成失配。同一Source产生Cloud和Shadow，支持不等于真实云影配对；平面/scene中心太阳角忽略地形、视角与云体视差。

保留率使用原完整A_Sh，precision使用共同定义的有效参考；同k不保证参考有效数相同。按scene汇总，不把patch或像元当独立重复，不做像元bootstrap。此简化筛选不是完整Fmask复现，也不构成新方法或泛化证明。

原14天窗口逐字继承，不重置。未满足全部两池门槛则停止当前候选，不改高度/角度/阈值、不自动进入下一phase。所有参数、选择数、失败与资源证据保留；完整备份及哈希验收完成后才交付。
