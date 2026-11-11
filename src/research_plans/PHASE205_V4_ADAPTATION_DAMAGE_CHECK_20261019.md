# Phase205：V4上是否仍出现适配后的重复场景损伤

## 只做这些

复用Phase200锁定的官方V4两权重，训练两个固定seed的RegNet decoder适配模型；同时评价单成员和完整双模型输出。**先确认V4有没有该现象，不追加BN交换、风险探针或修复方法。**

本文件仅立项，尚未构建模型、训练或评价。核验门通过后执行者按本协议继续，不因模型结构不同临时换成V1。

## 1．身份与数据核验，先交锁再训练

V4是SMP Unet，两个成员来自官方索引：

| 记号 | 官方encoder标识 | 权重SHA256 |
|---|---|---|
| R4 | tu-regnety_004 | 7f6e4202e17ee73efa4aba7abb5c34f4f90a9f7eb42480820714994dff2db660 |
| N4 | tu-edgenext_small | d5fe67ad00f6fdb73eb8382ad925849e97fb82cedb46225344afff1a734a9c1d |

- 以`outputs/phase200/phase200_20261017/input_model_lock.json`为入口，HF revision为`c9a4fb88188709127aa25cfe51ae7fd41b0132f8`；优先在受控shared复用Phase200的实际权重及已验收runtime，逐字节核SHA，不把本地索引当实际文件核验。
- 固定模型version=4、model_library=smp，不能仅用软件包版本号代表模型V4。源码沿用Phase200验收版本并锁SHA，不随远端main更新。记录Torch／SMP／timm和官方归一化、拼接、后处理源码身份。
- 3通道B04／B03／B8A、4输出类别Clear／ThickCloud／ThinCloud／Shadow；按实际state_dict形状和作者实现核对，不能从文件名猜类别数。
- 复用原100个fit相关组及Phase122／123各自1000步采样、裁剪、翻转序列。第一轮V4训练严格复用原seed12201的序列，第二轮复用12301的序列；优化器随机seed也分别12201／12301。训练前封存两份SHA，不根据V4表现重采样。
- 评价原Catalogue32组和PixBox16产品／10组；原53校准组不读取，本轮不校准。原65a／65b／65c、ALCD、Phase200案例不新增评价。
- 元数据核对V4公开训练覆盖：CloudSEN12 High／Scribble／2k／PC／SR、KappaSet；已有产品／footprint交集记录逐组引用，未核到的标unknown。OCM hard negatives／scribble未完整公开，不能宣称已排除所有交集。这里只复盘固定历史场景，不称独立域外验证，不按旧损伤或元数据结果挑组。
- 所有策略同一有效支持；PixBox8190／8188差异按本轮实际交集报告，先保存支持锁。不新下载影像／标签或读取未知新池。

官方核验入口：[版本及训练数据](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/MODEL_CHANGELOG.md)、[模型索引](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/omnicloudmask/model_download_links.csv)。作者自有部分数据未公开的说明见[方法页](https://omnicloudmask.readthedocs.io/en/latest/how-it-works.html)。

## 2．结构与实现门

- 按SMP真实模块识别`encoder`、`decoder`、`segmentation_head`；encoder参数及buffer冻结，decoder与segmentation_head全部参数可训练，BN运行统计允许随训练更新。若有其他参数／buffer先分类并落锁，不能默默漏训或多训。
- 记录每模块参数量、可训练参数占比、BN模块／仿射参数／运行buffer清单。**不套用V1的46参数张量、5个BN或14.14M可训练参数。**
- 官方loader与直接重建的原R4在同一归一化512裁剪上输出最大差≤1e-5；原双模型E4按官方链路推理，独立复算一次双成员合成／拼接／softmax及有效支持，差≤1e-5。
- 原V4集成公式按锁定源码执行：先均值双成员raw输出，再沿原patch坐标加权拼接及官方后处理。不得改成成员概率平均、平均最终Shadow分数或把一个新成员复制成两个。
- 单步梯度验收确认decoder／head有有限非零梯度，encoder无梯度、buffer不变；门检后恢复全部初始模型状态（包括decoder BN）、清梯度／优化器再正式训练，门检不能算第一步。
- 若结构、加载或初始一致性不过，停止训练，交付缺项与原因；不现场换架构、类别、尺寸或精度。

## 3．只训练两次，每次1000步

每seed独立从原R4开始；encoder eval且始终冻结，decoder／head train。N4全程冻结，不参与训练、不蒸馏。

- 原100组、原512×512裁剪／水平翻转、逐裁剪作者channel_norm、batch1、float32；不加新增强、采样平衡或光谱通道。
- AdamW：lr=1e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8，恒定学习率。
- 损失完全沿用C100：4类raw logits合成三父类，Clear原值、Cloud为Thick／Thin的logsumexp、Shadow原值；忽略原无效参考，无类别权重的交叉熵。
- 只用1000步末步，不早停、不选checkpoint，不看评价侧损伤改变训练。保存初始和末步、采样序列、loss、参数／buffer白名单与encoder不变回执。
- 两seed末步及训练回执封存后，才评价适配输出。不得追加第三seed、延长步数或增加“冻结BN”训练臂。

## 4．一次评价六个策略，分两组比较

20m、作者1000／300滑窗、float32、batch1、原归一化与拼接。复用同一N4的raw patch输出以节省调用，但必须按原公式重建集成；不要仅从成员最终confidence反推集成。

| 策略 | 内容 | 比较对象 |
|---|---|---|
| R4 | 原RegNet成员 | 成员基线 |
| C4-12201 | 适配RegNet seed12201 | R4 |
| C4-12301 | 适配RegNet seed12301 | R4 |
| E4 | 原R4＋原N4 | 完整双模型基线 |
| E4-12201 | C4-12201＋原N4 | E4 |
| E4-12301 | C4-12301＋原N4 | E4 |

这是**只适配一个成员的V4双模型方案**，不是同时微调整个双模型。原N4与E4身份、raw集成及最终分数均保留。

- AP取作者最终Shadow confidence，宏／池化与完整逐组同时列；原组有正负支持才算AP，无影组AP=NA但保留误报统计。
- 决策指标统一用官方四类argmax后合并Thick／Thin为Cloud，报告Shadow召回、非Shadow误报；不得混用三类概率合并后argmax或V1校准阈值。
- Catalogue另列池化三类mIoU及其缺类定义；PixBox不算全景mIoU／FPR。无阈值扫描、oracle召回或1%预算判定。
- **主要观察：每池相对原完整E4，两seed是否在相同组都出现ΔAP<−1个百分点。** 成员相对R4同样报告。全部组改善／稳定／损伤并列，不只列旧四组。
- 预先指定旧四组32TML／40XDR／19FDV／54TWN为历史跟踪，分别回答V4是否仍受损、是否仍跨seed、是否传递到集成。保留54TWN少量正参考限制；不同成员／版本／集成不能混算为一次重复。
- 每seed、每池报告宏AP差及配对相关组bootstrap10000次、seed205019，仅为描述，AP组数仍按实际7／8左右计算，不把seed或大量像元当新场景。

## 5．判断与交付

本轮没有“必须证明V4也会受损”的目标，也不要求所有指标同时获胜。

- 成员和集成都有两seed重复损伤：可讨论该固定V4适配设置的有限现象，之后再判断是否值得机制对照。
- 仅成员受损、集成未出现重复损伤：写明集成结果不同，不宣布完整Omni有问题。
- 未观察到重复损伤：保留结果，停止将V1的机制／预警线推广到V4；不放宽损伤定义或换seed追正例。
- 平均改善与局部损伤可以同时存在；−1个百分点是描述分类，不是显著性检验或机制证据。两seed也不估计所有训练随机性的失败概率。
- 若历史四组未复现，但其他固定组有重复损伤，完整列出，不重新定义“同一现象”来冒充旧地物机制复现。结构／数据／decoder容量等跨版本差异都保留。

输出`outputs/phase205/phase205_20261019/`，小结果摘要至`src/research_results/phase205_20261019/`。

**表1：每池六策略**

| 池 | 策略 | 全部组／AP组 | 宏AP | 池化AP | 官方决策宏召回 | 宏误报 | Catalogue池化mIoU |
|---|---|---|---|---|---|---|---|

**表2：全组重复损伤，成员和集成分表**

| 池／组 | 正参考数 | 原基线AP | seed12201 ΔAP | seed12301 ΔAP | 两seed均下降>1pp？ | 两seed召回差／误报差 |
|---|---|---|---|---|---|---|

另交V4身份／训练覆盖状态表、SMP结构／BN白名单、两seed训练与初末步锁、CPU独立AP／混淆／集成复算、完整支持损失、实际调用与资源账本、SHA。大权重／分数不入Git，不自动push。

README两页内只回答：V4成员是否复现；完整双模型是否复现；旧四组与其他组分别怎样；还不能得出什么。没有现象时如实结束本轮，不自动追加BN交换或风险检测。

## 资源

复用已有V4权重及环境；零新影像／权重下载、零环境升级。先核验原件可用与剩余空间。两次1000步训练＋固定评价＋工程验收全部计账，网络调用总上限8000次、GPU阶段≤4小时，GPU≤20GiB、CPU≤16GiB、新持久文件≤6GB、宿主物理盘余量≥4GB。原文件不改。

预算内工程恢复保留失败与调用；不得超限后只报主调用。无法在预算内完成则交付现状／缺项，不偷减评价组、换精度或另训模型。**本文件不执行训练；执行者在身份、实现与资源门通过后按以上固定内容运行。**
