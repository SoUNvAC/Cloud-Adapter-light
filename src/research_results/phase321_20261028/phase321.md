# Phase321：Source来源重叠与零训练敏感性分析

## 目标与范围

一次性核对Source官方train8490、选模val535与Catalogue513、ALCD37、PixBox29。V4降为参考，本轮未审OCM1751及完整V4来源，不为R4原指标签发外部独立性证明。Source的DINO预训练覆盖与历史目标开发暴露仍未知。官方train名单是保守的可采样全集，未按4000步重建实际采样清单。

## 改动与事先规则

同tile同UTC日列同景候选，精确时刻另列；同tile异日列地理重复；原多边形在同一目标UTM投影相交严格>1平方米列空间重叠，无缓冲。ALCD使用含ignore区域的完整影像网格，PixBox使用整幅母影像网格，所以空间排除相对实际标注/采样点保守。同景候选不自动证明裁剪像元泄漏。三种flag可同时成立，不能把列数相加当总数。

scene_clean排除同景候选和未知；strict_clean再排除同tile异日或空间重叠。train/val分别统计，敏感性另给两者联合排除。5%为预设同景候选占比小/大的分界。

## 核对结果

|目标|Source角色|产品数|同景候选|同tile异日|空间重叠|未知|scene_clean|strict_clean|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|Catalogue|train|513|0|30|6|0|513|482|
|Catalogue|val|513|0|1|0|0|513|512|
|ALCD|train|37|0|7|7|0|37|30|
|ALCD|val|37|0|4|4|0|37|33|
|PixBox|train|29|0|4|4|0|29|25|
|PixBox|val|29|0|0|0|0|29|29|

以上分母为完整目标池；具体已评价样本见TARGET_OVERLAP.csv的evaluated列，不能将完整池重叠比例套到已评价子集。每个命中对、面积及两侧面积比例均见MATCH_PAIRS.csv。

train+val联合排除（按产品去重）：

- Catalogue：同景候选0，同tile异日31，空间重叠6，strict_clean 481/513；已评价32产品中排除0。
- ALCD：同景候选0，同tile异日11，空间重叠11，strict_clean 26/37；已评价0产品中排除0。
- PixBox：同景候选0，同tile异日4，空间重叠4，strict_clean 25/29；已评价16产品中排除0。

## 同一批预测的敏感性

|目标|排除Source角色|子集|保留相关组|组均AP %|组均mIoU %|池化mIoU %|组均FPR %|
|---|---|---|---:|---:|---:|---:|---:|
|Catalogue|train|full|32|80.205602|61.279967|71.688882|3.957060|
|PixBox|train|full|10|45.884981|NA|NA|NA|
|Catalogue|train|scene_clean|32|80.205602|61.279967|71.688882|3.957060|
|PixBox|train|scene_clean|10|45.884981|NA|NA|NA|
|Catalogue|train|strict_clean|32|80.205602|61.279967|71.688882|3.957060|
|PixBox|train|strict_clean|10|45.884981|NA|NA|NA|
|Catalogue|train+val|full|32|80.205602|61.279967|71.688882|3.957060|
|PixBox|train+val|full|10|45.884981|NA|NA|NA|
|Catalogue|train+val|scene_clean|32|80.205602|61.279967|71.688882|3.957060|
|PixBox|train+val|scene_clean|10|45.884981|NA|NA|NA|
|Catalogue|train+val|strict_clean|32|80.205602|61.279967|71.688882|3.957060|
|PixBox|train+val|strict_clean|10|45.884981|NA|NA|NA|

Catalogue复用Phase317 Source32逐景混淆和统计，全集与原指标在1e-6个百分点内复现；PixBox复用Phase112 Source16产品/10相关组的统计，部分成员被排除时整组剔除。缺少充分统计的池化AP/FPR不重建，不将组均AP当池化AP。ALCD没有已冻结Source预测，性能NA。

## 网络、止损、失败与备份

本地GitHub SSH443推送成功；远端沿用已知GitHub TLS不稳定的既有路线，以逐SHA验证bundle执行真实git pull，未改origin。首次合成自检因GDAL已返回MultiPolygon而断言失败，尚未读取Source名单或计算真实匹配；原记录保留，独立recovery01只修几何类型处理。规则、面积阈值及池未变，实际匹配仅一次。
独立恢复实际CPU墙钟14.770秒；峰值RSS 175366144字节。CPU墙钟上限1800秒、进程地址空间8GiB、宿主空闲4GiB、双机新增200MB。零训练/模型加载/前向/像元读取/下载。所有原输入、执行代码及全部回传成员SHA通过；详见INPUT_FINAL_SHA、ACCEPTANCE及BACKUP_RECEIPT。

## 如何使用

仅可将无命中的已评价子集描述为“相对已审Source train/val未检出来源重叠的历史目标评价”；这不恢复盲测身份，也不认证骨干预训练独立。若某子集存在同景候选，完整集与去除后结果并列，来源范围写明。V4完整训练集合未审，R4 Catalogue82.14及既有PixBox结果保留为历史参考，不能用Source无重叠给V4背书。OCM hard-negative若与目标重叠，首先影响误报改善的解释，本轮对此保持未知。Omni教师版本选择不得以本轮Source审计替代其训练覆盖核查。

本轮结算，不启动下一phase。
