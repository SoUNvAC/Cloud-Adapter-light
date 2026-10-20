# Phase302：VFM固定伪标签适配的有向错误转移

## 一、做什么

复用已有DINOv2-S＋Cloud-Adapter三父类Source，在Catalogue原100个fit代表图像上做一次固定伪标签LoRA适配。固定单seed，对照Source-only与Source+LoRA-PL，检验：Cloud→Shadow混淆是否减少，同时新增Surface→Cloud错误；是否存在原正确Surface预测被改判Cloud的损伤。

本轮完成输入/模型核验、一次训练、固定评价、错误流和结果整理。只用无标签目标图像适配，不访问源样本，按SFDA记录。不重训Source，不加入M/B比较、可靠性约束、额外seed或新数据。不是Phase64/65监督分枝续训。

## 二、怎么做

### 1．固定Source与训练接口

- Source：`src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth`，SHA256必须为 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。本地缺文件时核对远端既有原件，不换成Phase22四类模型、旧目标适配端点或Omni。
- 模型配置由 `src/configs/protocol/phase64_source_parent_rgb.py` 及其继承链展开，保存resolved配置、DINO有效block、Cloud-Adapter、Mask2Former头和类别顺序。三类固定Surface=0、Cloud=1、Shadow=2，无Thin/Thick临时合并。
- 在前向前核对Source训练/选模记录与已存CloudSEN12→Catalogue产品/空间审计，写明训练、验证来源、目标历史使用及未解决交集。核验不读取源影像/标签。不能仅凭checkpoint名把它称为严格独立Source；已知训练污染或权重身份不符时先交缺项，不改选模型。
- 复用 `Phase64LoRACloudAdapterDinoVisionTransformer` 和 `LoRALinear`，rank=4、alpha=4、dropout=0，注入DINO线性层qkv/proj/fc1/fc2；不使用Reins token低秩模块冒充LoRA。只优化实际参与前向的LoRA A/B，保存注入层/有效block清单及机器统计参数量。
- DINO原权重、已有Cloud-Adapter、neck、分割头、全部原buffer冻结，始终使用eval行为。LoRA B零初始化；初始化/关闭LoRA均应复现Source，同一输入最大绝对差<=1e-6。严格核对所有基础张量和加载键，缺键仅允许新增LoRA项。
- 使用可穿过冻结head/neck/backbone的求导路径；不能沿用Source的 `FrozenBackboneEncoderDecoder.extract_feat` 中整体no_grad/detach。可复用FrozenHead外壳，但须验证梯度实际进入LoRA。原模型eval后单独设置LoRA白名单requires_grad，避免其train(False)再次冻结LoRA；后续模式切换均重新验收。

### 2．固定数据、输入和窗口

- fit：Phase300的 `FIT_IMAGES.json` 中原100组代表；evaluation：原Catalogue32 development代表。只读这些产品，不访问53校准组、65a/b/c、ALCD或旧sealed。本轮不运行PixBox；先完成VFM机制初筛，PixBox另立完整RGB输入与成本协议。
- 训练manifest只保留图像身份，不含标签路径/摘要、common_valid、类别比例。新建纯影像RGB加载路径；不能调用仍隐式读取masks.zip的 `catalog_image(read_truth=False)`，也不能直接用含mask的Phase65 prepared dataset。
- Catalogue原float32 TOA的RGB索引[3,2,1]，乘255匹配既有Source数值尺度，不截断>1反射率，不用缩略图、百分位拉伸或Omni channel_norm。按Source既有BGR输入约定交给其bgr_to_rgb预处理，避免双重交换；保留Source均值/方差及padding配置，锁定一次预处理前后数组SHA。
- 训练有效支持只由三输入有限、非零、非原始float32填充值65535/10000决定；无效输入填零。评价采用Phase300封存共同参考支持，并核对新RGB输入有效支持；如有损失，先列数量和原因，不暗删像元或更改旧共同支持后宣称同支持。
- 教师和两策略评价统一在原20m的1022格网上用512窗口、步长256，末端补齐到图像边缘：每轴起点[0,256,510]，每产品9窗。每窗走同一Source数据预处理/模型，恢复512像元得分后按逐像元等权平均拼接，不先argmax再投票，不整景缩成512，不套用Omni1000/300拼接。
- 核对网络内部pad/裁回及Mask2Former上采样与既有模型实现一致。完整无效窗口也保留固定9窗调度，像元支持在输出后取交集；全部实际窗口入锁。
- 新Source/学生均重算，旧Phase65整景缩放缓存与Omni缓存不复用。RGB、Source强弱、结构和视野都不同，不将Phase300与本轮差异解释为单一骨干因果效应。

### 3．生成固定伪标签与训练

1. 只在100个fit图像上用Source生成900次窗口预测。Mask2Former输出按原头实现计算三类语义得分 `s_c=sum_q softmax(class_q)_c * sigmoid(mask_q)`，排除no-object；记录该量是聚合得分而非归一化概率，不能误当原生分类logit。
2. 先平均窗口三类得分，再沿类别计算 `p=softmax(s)`；与旧Source评分口径一致，但不是校准概率。argmax与s一致，最大p>=0.90的图像有效像元保留三类硬伪标签，其余ignore=255。固定阈值，不看标签调覆盖。保存三类得分/伪标签/最大p/valid及SHA，可仅为4个探针产品保留完整三类得分，其余保留伪标签和最大p以节省空间。
3. 训练随机seed=30201；LoRA初始化seed=30201，记录初始化后再重置训练随机流。裁剪产品/坐标/翻转复用Phase300已锁定的1000步 `CROP_SCHEDULE.json`，核对product身份和顺序；只是几何采样配对，不沿用Omni伪标签、波段或归一化。
4. 一次训练，1000个尝试步骤，batch1、512裁剪、FP32、仅水平翻转；AdamW lr=1e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8、恒定学习率。无颜色增强、EMA、类别重采样、阈值刷新或额外loss。
5. 取Mask2Former最终query层的class/mask输出，用与教师相同的可微语义聚合得分s，计算接受像元的 `CE(s, pseudo)`，按接受数平均。不调用原监督Hungarian matching/原mask loss读取真值；不把旧head的predict/test_step外壳中的no_grad带进学生训练路径。验收聚合得分与原head预测输出一致且LoRA收到梯度。
6. 全ignore裁剪跳过前向和optimizer更新，保留尝试步及实际更新数；不补步、不仅做weight decay。只保存初始和第1000尝试步末权重，不早停、不按development选checkpoint。
7. 同512探针取Phase300原4个fit产品的[255:767,255:767]中心裁剪。在第0/100/500/1000尝试步各预测一次，共16前向；第0步零LoRA必须复现严格Source同裁剪，至少一裁剪额外关闭LoRA复核。保存三类s、预测、valid及SHA；完整教师在同位置的预测另存。末步封存后才连接探针真值。
8. 原参数/buffer初末SHA一致，优化器仅含LoRA；记录实际更新、损失和梯度。伪标签/训练进程不得访问目标标签或development图像；本地读取真值的统计程序在封存后运行。历史已用标签和开发事实照实披露。

### 4．固定评价和错误流

- 末步权重SHA锁定后，在原32组按相同9窗协议各运行Source和LoRA-PL，共576次窗口前向。保存三类s、Shadow评分softmax(s)[2]、三类argmax、有效索引和支持SHA。Source与学生评分/读出严格一致，不套用Omni四类argmax或四到三类合并。
- 主要报告两项有向错误：Cloud→Shadow和Surface→Cloud的净变化及逐像元修正/新增；所有六种有向错误和全部27格 `T[truth,Source_pred,student_pred]` 完整保留，边缘应精确复现两混淆矩阵。
- 同时报各类IoU、池化/组均mIoU、Shadow宏/池化AP、宏召回/误报、最大组误报。缺类规则沿用Phase300并明确表注，保留全部32组；不因3像元缺类惩罚改主指标，不将单个组主导的均值变化当全域现象。
- 复用Phase301参考云边界邻域定义、全部组及排除规则，报告两项错误在边界/非边界分布及单组集中度。不用边界邻域代替薄云标签，不根据误差挑少数景。
- 四个探针分别比较严格Source同512/零LoRA与末步；按真值类报告纠错、恶化和净正确数，另列完整教师→初始/末步的对照。不得将上下文不一致的净纠错当成训练收益。
- 独立CPU重算混淆、错误流恒等式、AP和组均/池化指标；报告单seed描述性差值，不追加显著性检验、阈值扫描或独立确认结论。

## 三、交付什么

输出至 `outputs/phase302/phase302_YYYYMMDD/`，摘要至 `src/research_results/phase302_YYYYMMDD/`；日期为实际开始北京时间。报告 `RESULT.md` 两页内，分“数据结果、结论分析”。

**表一：Source身份与执行。** resolved配置、实际文件SHA和审计证据另附。

| Source权重SHA | 骨干/有效blocks | 训练/选模来源 | 目标交集已知/未知 | 输入/预处理 | 语义得分定义 | 原/新增/可训练参数 | 实际训练runtime |
|---|---|---|---|---|---|---|---|

**表二：固定两策略结果。** Source-only、LoRA-PL两行及一行差值；指标%与差值pp分开。支持写全部组/AP有效组数。

| 策略 | 支持组数 | 组均/池化mIoU | Surface/Cloud/Shadow池化IoU | Shadow宏/池化AP | 宏召回 | 宏误报 | 最大组误报 |
|---|---|---|---|---|---|---|---|

**表三：有向错误流。** 六类错误均有行，27格及逐组表另交。

| 错误类型 | 真值支持 | Source错误数 | 学生错误数 | 消除旧错误 | 新增同类错误 | 净减少 | 消除中改正/转为另一错误 | 错误率差pp |
|---|---|---|---|---|---|---|---|---|

**表四：错误位置与集中度。** 全组逐项保留，边界与非边界给各真值类支持及独立排除数量；最大的损伤组附ID但不能替代全表。

| 区域/相关组 | 真值类支持 | 错误类型 | 消除/新增数 | 净变化 | 对应错误率差pp | 占全池同类新增/净变化比例 |
|---|---|---|---|---|---|---|

**表五：同512训练变化。** 四探针×三个真值类，报告所有时点；完整教师对照另交，不混入本表。

| 产品 | 时点 | 真值类 | 支持 | Source正确→学生错 | Source错→学生正确 | 净正确变化 | Shadow预测比例 |
|---|---|---|---|---|---|---|---|

另交伪标签三类覆盖/逐产品表、1000尝试/更新/跳过数、训练曲线、全部32组混淆及指标、纯影像访问验收、输入/权重/初始化/代码/裁剪/伪标签/端点SHA、梯度/冻结/读出一致性检查、独立CPU验收、真实调用/时间/峰值内存与失败账本。

报告直接回答：VFM上是否同样出现Cloud→Shadow减少而Surface→Cloud增加；新增错误是否集中于非边界或少数组；相同上下文下训练真正改正了什么、损伤了什么。若不复现，报告与Omni设置的差异，不硬套旧机制；若复现，再提出一项针对有向错误的约束实验，不直接扩成三seed结构竞赛。

执行预算：计划900教师＋最多1000训练＋576评价＋16探针=2492前向；另留初始化一致性/求导验收、失败及恢复余量，总新增网络前向<=4000次、GPU阶段<=4小时。按顶层网络或decode路径完整计算一次计一次，不能绕过计账；构建中实际前向也计。GPU<=20GiB、CPU<=16GiB、新持久文件<=6GB、宿主空闲>=4GB；一次一个GPU任务，先检查冲突。

复用既有 `cloud-lite-pt210` VFM环境并核验mmcv/mmseg算子；Phase300的Omni历史runtime许可不能当成本轮换环境依据。未解决兼容性或预计超预算时交具体缺项，不自动升级、改精度、缩窗口或追加资源。无新影像/权重下载，不读取未列评价池，不自动push、定时检查或启动下一phase。本文件只立项，不启动运行。
