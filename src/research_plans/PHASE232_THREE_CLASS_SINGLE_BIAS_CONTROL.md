# Phase232：三类联合单Cloud偏置对照

## 做什么

复用Phase231的模型、60景校准池、10景validation与一个Cloud偏置参数，只把二类拟合改为**三类联合交叉熵**。比较能否减少地表误报和云影牺牲，并完整报告放弃的云召回／mIoU收益。不是新算法，不预设新损失一定更好。

零网络训练、零参数更新，不加第二个偏置、正则、类别权重或场景门控，不搜索新α／checkpoint。不读取任何test、L8或源域影像／标签。

## 1．模型、数据与补算

- 模型仅Source、L8-C21701、L8-C21702；原末头、α=0.25、插值／骨干／buffer／FP32／评价配置全部沿用230—231。
- 校准：231原60个SPARCS target_train场景，输入身份见 `phase232_frozen_20261027/CALIBRATION_INPUTS.json`；整图resize512、不裁剪增强，原三父类映射。
- validation：只读230原10景分数／标签缓存，禁止新前向；231的二类偏置只读复用，不重拟合。
- 231仅存Cloud／Surface差d2，不能恢复三类联合统计量。需对3模型×60景补180次前向，不擅自用d2近似第三类。
- 补算每张图时重算原float32 d2并与231缓存逐值相同，标签与231缓存一致；身份／输出漂移就停止，不带着漂移比较。
- 输出仍为原非负归一化语义分数p，不再softmax，不称为原始logits或已校准概率。只保存以下d3充分统计量和每景损失常数，不保存全三通道／骨干特征。

## 2．唯一拟合规则

全部有效参考Surface／Cloud／Shadow都参与，ignore排除；场景等权、景内像元等权，不平衡类别、不抽样。令k=Cloud，其他为Surface和Shadow：

```text
s_j = log(max(p_j, 1e-12))   # 用float64计算
L3(b) = mean_scene mean_valid [logsumexp(sSurface, sCloud+b, sShadow)
                              - s_reference - b*I(reference=Cloud)]
d3 = sCloud - logaddexp(sSurface, sShadow)
g3(b) = mean_scene mean_valid [sigmoid(d3+b) - I(reference=Cloud)]
```

同一模型只拟合一个b，范围仍[-5,5]，沿用231导数单调二分60次、边界规则与float64实现。不能因为结果不好扩边界或换系数。该参数化下，三类CE关于b的优化等价于Cloud vs nonCloud的BCE；**它不独立调整Surface与Shadow边界**，不要写成三个自由参数。

缓存每景d3为float64、复用231标签；另保存每景 `K=L3(0)-BCE_d3(0)`，即可准确重建 `L3(b)=K+BCE_d3(b)`。不得把BCE绝对值冒称三类CE绝对值。直接小数组核对两种计算的loss差与导数相同，并核对g3的数值差分。

相对231，分母与有效支持都变成三类联合定义；这不是只给旧损失简单追加一批Shadow像元的单因素分离实验，解释时保持边界。

**先封存三个b3、校准支持、损失／导数和输入／代码SHA，之后才能读取validation缓存。** 60景不是独立于历史SPARCS头训练的新校准集，结果仍属探索；Source／L8头用了SPARCS标签，不能称无监督迁移。

## 3．固定评价与对照

每模型只有3种决策，共9行：

- Raw：b=0。
- B2：231原二类损失冻结b。
- B3：本轮三类损失冻结b。

全部采用同一决策 `argmax(pSurface, exp(b)*pCloud, pShadow)`，原父类顺序及ties规则。Raw／B2必须精确复现231逐景混淆；B3仅CPU作用于230缓存。不要读取或使用之前的val偏置网格来选b，不加oracle最优点。

### 表1：拟合

| 模型 | 60景三类支持 | b2→b3 | L3(0)→L3(b3) | g3(b3)／是否触边 |
|---|---|---|---|---|

### 表2：全部val结果

| 模型／Raw或B2或B3 | 池化／场景平均mIoU | Cloud召回 | Surface→Cloud FPR | Shadow召回 | 非Shadow→Shadow FPR |
|---|---|---|---|---|---|

召回与误报均交池化与场景均值，明确分母和有效场景数；缺类项NA。保留三类IoU、支持与逐景3×3混淆。

### 表3：预定主要比较B3−B2

| 模型／seed | ΔmIoU | ΔCloud召回 | ΔSurface→Cloud FPR | ΔShadow召回 | 原正确Shadow→Cloud减少量 |
|---|---|---|---|---|---|

Source单列，两L8 seed分别列及指标差均值，不做分数集成。表中总体指标以池化为主，完整CSV另交场景等权差。降低误报／保住Shadow若伴随Cloud收益减少，必须并列，不称全面改善。

逐景全部10景列B3−B2和B3−Raw。重点另列LC81500412014244LGN00（地表误报集中）、LC81490372013362LGN00（云影牺牲）、LC80380362013144LGN00（漏云恢复）；它们是既有事后诊断景，不是新确认样本。

保存同参考像元的Raw→B2、Raw→B3、B2→B3直接转移表，区分原正确Shadow的实际丢失与净计数差。核对固定常数偏置不改变Cloud／Surface的d排序，不宣称修复了排序。无需新bootstrap或显著性门槛。

## 4．预算与交付

最多 **220次完整输入前向、GPU30分钟**（主体180，其余仅构建／门检／固定重复，失败也计入）；validation新前向0。CPU拟合／分析最多60分钟，CPU8GiB、GPU20GiB、新增输出512MiB。180份d3 float64约360MiB，标签与旧缓存只读复用；不要额外复制旧全包或保存训练三通道分数。

输出 `outputs/phase232/<运行目录>/`：RESULT.md、上述CSV、全部逐景指标与转移、三个b3封存文件、d3与K回执／SHA、模型与输入／代码身份、调用和资源账本、独立CPU求解／指标复核。缺身份、数值不一致或达到预算就交已完成项与缺口，不自动重训或追加方案。

验收结论只回答：纳入Shadow的三类拟合，是否在这批validation中改善231的代价分配，以及付出了什么。若仍有局部损伤，不自动推出必须做场景自适应；若有效，也只确立一个简单基线。本聊天仅立协议，不启动远程任务。
