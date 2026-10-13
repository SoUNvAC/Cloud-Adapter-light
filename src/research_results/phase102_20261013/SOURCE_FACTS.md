# Phase102：固定来源事实与未核实字段

来源记录、原文件长度、官方MD5及API响应SHA逐项见inputs/source_records.json与source_access_sha256.json；不是大包校验通过记录。

|项|KappaSet7100327|PixBox5036991|
|---|---|---|
|固定来源|https://zenodo.org/records/7100327；DOI 10.5281/zenodo.7100327|https://zenodo.org/records/5036991；DOI 10.5281/zenodo.5036991|
|许可|API声明CC-BY-4.0|API声明CC-BY-4.0|
|官方大包|kappaset.zip：56,186,308,652 bytes，MD5 3f435c61efbb511c13a353a322787e42|Sentinel-2_L1C.zip：22,000,157,118 bytes，MD5 621c1185569e1976e12e7acc032273ce|
|本轮小资料|README.pdf 231,988 bytes，MD5 98bf16a2d90cba386b197133a24f402f；仅ZIP尾262144 bytes和目录1213411 bytes|PixBox-S2-CMIX.zip 1,537,886 bytes，MD5 c279032077e046518fa7e9618be9d81f；取得含标签CSV与定义TXT，ZIP MD5核对通过|
|人工／半自动流程|官方说明CVAT及Segments.ai，结合主动学习半自动标注，经过数据验证；不是逐像元可靠率认证|官方说明训练过的经验专家逐点分类；auto-associative neural network作质量检查；采样设计与逐点人工裁决一致性尚未核验|
|类别定义|0 UNDEFINED；1 CLEAR；2 CLOUD SHADOW；3 SEMI TRANSPARENT CLOUD；4 CLOUD；5 MISSING|SHADOW_ID：0 None of classes；1 None；2地形／树／建筑影；3云影；4云上影。CLOUD_CHARACTERISTICS等另列多属性，不能直接假设互斥三类映射|
|影像输入|L1C，13波段加Label；特征以Sinc IIR／Blackman滤波重采样为10m，512×512；六所需波段仅由说明声明，未开NetCDF|29个严格原SAFE前缀及PRODUCT_URI匹配；每产品B02/B03/B04/B08为原10m、B11/B12原20m，六对象均有长度／MD5／generation索引，未下载波段|
|TOA缩放／偏移|原处理基线／NetCDF存储缩放未核实，不能假定DN÷10000|本轮原产品XML的QUANTIFICATION=10000、旧处理基线02.04—02.06，无RADIO_ADD_OFFSET字段；special值NODATA=0、SATURATED=65535。后续读取仍须核验内容、无观测与预处理，不能混用重处理N05产品|
|位置精度|README给文件索引乘512，但无CRS/原点/轴向和原产品完整名；1079个键与官方1038产品未调和|WGS84点位、原像素索引、原XML UTM格网；筛查使用完整109.8km父格网加固定缓冲保守排除，不声称上下文或语义已验收|
|统计支持|挑战子块选择可能偏倚，不能把9251子块当独立组|选择性像元集合；未知包含概率／采样权重。可讨论固定样本条件下的标签一致性，不能推断完整场景FPR／密集mIoU|

## 谱系

5095024为旧KappaZeta资料（官方说明4403子块／155产品）；旧版0=MISSING、5=UNDEFINED，与7100327这两个编号的含义相反，不能混用标签映射。本轮仅取旧README及API核查谱系。

请求11121064 API解析到版本记录11121065（V1）；声明从7100327重分类9251张掩膜，四类0 clear／1 thick cloud／2 thin cloud／3 cloud shadow，并给出原生波段分辨率。它是衍生重分类，不是额外独立池；没有下载任何重分类波段／mask包。其“2020”说明也不能替代7100327目录中2020—2021键的逐项身份核验。

只通过KappaREADME所链接cm-vsm的公开代码／文件树了解导出背景，访问SHA在清单中；当前master不能证明当年导出版本或实际NetCDF量化。Source基础预训练语料未知，CloudSEN12衍生物不能自动视作独立。

## 范围内历史使用核查

inputs/exposure_scan.json列出657个实际扫描文件及SHA。未发现固定候选使用记录；未搜索旧sealed像元／预测／指标或work_dirs，不能外推为“任何人从未用过”。Phase86两份历史位置原始清单SHA与其既有artifact_hashes一致，来源链在history_positions_only.json保留；未打开历史栅格。
