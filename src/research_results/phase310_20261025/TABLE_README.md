# 表口径

- ERROR_FLOW/BOUNDARY_ERROR_FLOW及TRANSITIONS27复用原304字段名：Source_errors、Source_pred表示comparison箭头左端旧模型；student_errors、student_pred表示箭头右端模型。Q90-PL->Q90-SP行的Source字段实际指Q90-PL，不是Source-only。
- DAMAGE_BENEFIT的D严格为Source正确Surface且Q90-PL新变Cloud；全池506798，非G/P内450058。T25XEG全部D206076，P内197082，两者不混用。
- outside_D_new_Surface_to_Cloud限定Source正确Surface、原Q90-PL不为Cloud、本轮变Cloud；不等于所有新增错误。outside_B_new_SourceShadow_to_Cloud限定原SourceShadow真Cloud且旧Q90非Cloud，本轮Cloud；SourceWrong列另含SourceSurface真Cloud。
- PROTECTION_COST按原影像定义P/A/其余区域及真值类逐组报告，可核查P内真Cloud/Shadow误保护；不能把P视为真Surface或参考R。
- 同512 PROBES的Q90为相同尝试步缓存。全图评价的基线Q90为固定1000步末权重。固定前三诊断产品T25XEG/T40XDR/T35RPM在全部32组表中保留；不按本轮结果重新选择。
- 曲线是单seed原始逐尝试步轨迹；45个空A步骤为缺口，SP在P为空时为0。没有平滑、跨seed置信区间或由探针选模。
