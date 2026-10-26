# Phase317 实验记录

- 目标：三池Source/Q90-PL/同掩膜CB探索基线，判断简单均衡对Shadow的效应与代价；不是新方法/新盲测。
- 改动：CMC只新CB，SPARCS/L8各PL+CB；原Source全新30201 LoRA、同初始化/日程/接纳集合，冻结主干head/buffer。无下载/升级/调参/EMA/目标监督初始化。
- 输入：CMC原TOA/9窗；SPARCS原RGB预览、L8原RGB PNG固定512。L8仅shadow=yes，100fit来自18scene、963val来自8scene；训练不读mask，评价封存后从原mask恢复被PNG合并的fill并精确对齐。
- 网络：本地代码push→gzs pull；最终流式全量回传，故障如有保留独立记录；无新数据请求。
- 止损：新增前向≤11000、GPU≤12h、CPU≤16GiB、新持久含双机≤16GB、宿主空闲RAM≥4GiB/磁盘≥8GiB；不重训/补步/降精度/删val，不加seed/loss。
- 结果：5新臂均封存；前向8267、GPU1453.64秒；CB在2/3池ShadowIoU相对PL增加，逐池收益/地表误报/Cloud纠正撤销及支持不足见DECISION，不以均值掩盖失败。独立CPU原mask/27格/六向流/D/B/SHA核对通过，CMC旧D/B精确。
- 局限与决定：单seed且历史暴露，显示512与TOA协议不同；本轮建立普通均衡基线，不包装创新。无自动下一phase/定时检查；14天窗口不随等待/失败重置。
