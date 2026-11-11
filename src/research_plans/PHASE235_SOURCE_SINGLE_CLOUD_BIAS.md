# Phase235：Source单Cloud偏置基线

## 1．执行范围

冻结Source网络，只增加一个全局Cloud偏置。**零网络训练，仅L8新增一次标量拟合；SPARCS复用232参数。** 不给C头或插值点继续加偏置，不扫阈值／α，不增加方案。当前只执行235，不自动启动236。

固定身份与清单见 `phase235_frozen_20261027/`。Source与234相同，FP32、eval、原RGB加载和Mask2Former读出不变；不改骨干、头、buffer或分辨率。

| 目标 | 偏置拟合 | 评价 |
|---|---|---|
| L8 | 原222的1560裁块／18个train场景，按原生标签去Fill | 原234 val963／8景、test2167／16景，VALID支持 |
| SPARCS | 直接复用232的Source三类CE偏置 `b=1.5086972783056742`，不重拟合、不再读取60景训练像元 | 原234 val10景、test10景，原支持 |

只用Source一个模型；它没有本轮训练seed，不复制成两个独立实验。两seed只来自作为参照的234普通监督C头。

## 2．先核验标签和输入

- L8 train清单按 `new_split=target_train` 使用；旧文件夹名train/test不是本次划分，不能据目录名重分。核对1560个文件SHA、18个scene及与val/test的scene互斥。
- L8 train原生mask尚未在本协议绑定。只在已有L8原mask目录按固定scene定位，复用234已核验的裁块轴规则；每块须精确复现旧PNG：native 0／128→PNG0、64→1、192→2、255→3。再排除native0，保留128 Clear。记录原文件、native数组和valid支持SHA。不把PNG0全部ignore。
- 缺原mask、映射／SHA不一致、含未知类别或对齐不唯一而影响Fill支持，停止受影响目标并报告；不删除难例、不猜轴、不下载／补样／重建RGB。
- L8 val/test原生支持复用234已验收身份，逐图Raw混淆及可用的预测SHA必须精确复现234 Source VALID；SPARCS Raw同样复现234。Source在SPARCS val的偏置结果还须复现232的Source B3逐景混淆。
- 每个train场景有效像元、三类支持、Fill数量单列；无有效像元的裁块保留清单并跳过损失。若整个场景无有效像元，报告缺口，不静默减少18景。

## 3．L8唯一拟合规则

逐裁块按原评价方式运行Source，不随机裁剪／增强。将wrapper的非负三类语义分数按通道和归一化为p，不再softmax；类别顺序Surface、Cloud、Shadow。

只用有效train标签，**18场景等权，景内所有有效裁块像元等权**，不是1560裁块等权，也不按类别平衡。重叠裁块如原清单存在，保持原清单并披露重复支持，不擅自去重改池。

```text
s_j = log(max(p_j, 1e-12))                  # float64
d3  = sCloud - logaddexp(sSurface, sShadow)
L(b) = mean_scene mean_valid [logaddexp(0, d3+b) - I(y=Cloud)*(d3+b)]
g(b) = mean_scene mean_valid [sigmoid(d3+b) - I(y=Cloud)]
```

它与三类CE关于这个单Cloud偏置的优化等价；上式绝对损失只称BCE，不冒称完整三类CE。Scene内分母为全部有效像元数，不是Cloud像元数。

b限制为[-5,5]：若g(-5)≥0取-5；若g(5)≤0取5；否则稳定float64二分60次，取最终区间中点。沿用232的数值实现。不扩边界、不换损失、不按val挑b。有效支持须同时有Cloud与非Cloud，否则报告失败。

按块保存必要d3／标签，禁止缓存全骨干特征或反复前向来求导。交独立CPU求解回执、L(0)、L(b)、g(b)、边界状态；小数组核对三类CE与上述损失的导数一致。

**先封存L8的b、SPARCS复用b及来源SHA、拟合支持和代码SHA，再评价偏置后的val/test。** 两池历史test已暴露，本轮仍属探索性复核。不得声称零目标标签或独立确认。

## 4．固定评价和比较

每目标只新增一种决策Source+B：`argmax(pSurface, exp(b)*pCloud, pShadow)`，沿用原类序和ties规则。b=0须精确复现Raw。分块稳定计算；不再softmax，不用val重选偏置。

所有参照只读复用234同支持结果：Source、两seed各自val选定W/P、两seed末头C。不得换成test网格最优α。L8主表使用VALID，OLD不进入正式排名；完整16景仍是主分析，yes六景／no十景另列，no不代表真实无云影。

输出 `outputs/phase235/<运行目录>/`，RESULT.md只需以下表格和简短结论：

**表1：拟合与标签预算**

| 目标 | train场景／裁块 | 有效三类像元／Fill | b | L(0)→L(b)／g(b) | 新拟合或复用 | 拟合／val选模标签预算 |
|---|---|---|---|---|---|---|

SPARCS列原60景拟合支持及232回执；省掉本轮拟合不等于没用这些标签。Source+B不使用val选b，W/P各使用8／10景val选α，分别记账；有效标签像元与独立场景数均列，不能将裁块数当独立样本量。

**表2：原val和test全部策略**

| 目标／split／策略／seed | 池化mIoU | 场景平均mIoU | 三类IoU | Cloud召回 | Surface→Cloud FPR | Shadow召回 | 非Shadow→Shadow FPR |
|---|---|---|---|---|---|---|---|

保留逐景3×3混淆与有效类数；IoU按union>0平均，召回无正类则NA。召回／FPR同时交池化和场景均值及有效场景数。FPR分母分别为参考Surface、参考Surface+Cloud，不能混称整体误报。L8 no层的Shadow误报仅为名义参考口径。

**表3：固定test主要差值**

| 目标 | Source+B−Source | Source+B−val选定W | Source+B−C末头 |
|---|---|---|---|

报池化mIoU差及95%描述区间；后两项分别列两个seed和差值均值，不集成分数。按原16／10个scene配对bootstrap10000次，seed23501；两seed共用抽样索引，每次聚合混淆后计算差值再取seed均值，Source+B只作为同一个参照。b和α全程固定，区间条件于已拟合／选定策略。P的对应差只交点估计，不追加有利检验。另交全部逐景差及L8 yes／no分层点估计。

## 5．预算、验收与交付

- 主体最多 **4710次完整输入前向**：L8 train1560＋L8 val/test3130＋SPARCS val/test20。若SPARCS val复用230已封存三类缓存，则少10次；234仅有混淆不能代替本轮偏置所需分数。
- 总上限 **4800次、GPU1小时**，包含探测、重复、失败；CPU拟合／分析4小时、CPU16GiB、GPU20GiB。只运行Source，不重跑W/P/C。
- 远端临时d3／标签缓存最多6GiB，按块读写，不复制历史全包；新增正式交付最多512MiB，不传回全池分数／特征。保留缓存路径、格式、SHA和支持量以便复核，清理只针对本轮明确生成的临时缓存，不动历史文件。
- 交RESULT.md、三表CSV、逐图／逐景混淆与预测SHA、L8 train原生mask绑定和支持、偏置封存文件、模型／输入／代码SHA、调用／资源及失败账本、独立CPU参数／指标／bootstrap复核。核对每种决策混淆总支持一致。
- 数值漂移、缺输入或达到预算，交已完成项和缺口，不自动重跑、扩大预算或开启新训练。

结论只回答：一个训练侧全局Cloud偏置能获得多少目标收益，伴随哪些Cloud／Surface／Shadow代价。没有收益也照实提交；本轮不证明偏移物理成因，不要求显著性通过，不据此自动启动新模块。
