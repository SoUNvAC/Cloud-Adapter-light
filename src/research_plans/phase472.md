# Phase472：MsRE主路径邻域遮光对比投影

- 做什么：B在原MsRE投影输入中加入x-reference对比。8个间隔2-token邻居，用可学习语义相似度、色度接近度和较亮偏好选软参照。直接通过原投影层处理，无零初始化输出头。
- 对照：A原网络同范围续训；B增加上述结构。两组均从450起步，只开放原MsRE projector和解码器；其余原网络、450分支、BN及DINOv3冻结。B另训练6147参数参照选择器。
- 怎么做：CloudSEN训练8490/原始验证535，目标监督0%，固定argmax。各6000步，batch4、seed45201；原模块lr1e-5，新模块1e-4。B强度0→0.25，2000步达到并固定；两组仅在2000步及以后选训练最优点，step0作为回退。
- 初始强度0精确复现450；强度写入checkpoint。推理不读取标签。RGB描述不是物理反射率，“相似参照”不等于已知同材质。
- 验证：535图独立评测、107ROI配对区间、漏检/误报；B同权重关闭对比、只关闭颜色与较亮偏好。记录4个阶段实际投影输入变化，检查主路径确实参与。
- 交付：主表、RESULT.md、代码、权重和签名。单seed6000步结构续训，不称40k从头重训；目标云影70%且地表/云不下降。

参考：[Adaptive Illumination Mapping for Shadow Detection in Raw Images, ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Adaptive_Illumination_Mapping_for_Shadow_Detection_in_Raw_Images_ICCV_2023_paper.html)中的多尺度对比动机。本轮不使用RAW照明映射，实验实现为RGB/语义引导的MsRE原投影输入对比。
