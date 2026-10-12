# Phase93：非线性光谱基线与Source分数增量

## 做什么

Phase92显示Source＋光谱优于纯光谱逻辑回归，尚不能排除线性基线表达能力不足。本轮固定两个相同配置的直方图梯度提升树：

| 模型 | 输入顺序 |
|---|---|
| G6 | B04、B03、B02、B08、B11、B12 |
| G7 | 同上六列，再加原Source云影概率的logit |

**只做两次树模型拟合；零分割网络训练、零网络前向、零遥感数据下载。**主要比较G7−G6，不搜索树参数，不根据开发／ALCD结果增减树数。

## 1．样本与固定实现

- 原153个fit代表，严格复用Phase92的抽样索引、Shadow二分类标签和有效mask。全部波段为原生TOA；G7分数复用原Source缓存，clip／logit按Phase67。无标准化、额外指数、局部特征或类别平衡。
- 保留组等权：若总样本数N、组数G、组i抽样数n_i，每点`sample_weight=N/(G*n_i)`，权重和为N。不能直接用和为1的旧权重，以免改变树模型L2项的相对尺度。
- 固定scikit-learn 1.5.2的`HistGradientBoostingClassifier`：

```python
loss='log_loss'
learning_rate=0.05
max_iter=200
max_leaf_nodes=15
max_depth=4
min_samples_leaf=20
l2_regularization=1.0
max_bins=255
max_features=1.0
categorical_features=None
monotonic_cst=None
interaction_cst=None
class_weight=None
early_stopping=False
warm_start=False
random_state=93012
```

- 不自动划随机像元验证集；两模型均跑满200轮，不看中间轮次选模型。最多4个CPU线程，记录版本、实际参数、轮数、时间和峰值内存。
- 既有conda可能没有sklearn。可使用匹配版本的已有环境，或在项目独立Phase93虚拟环境安装固定版本并锁定依赖；不得修改原推理conda。环境不可用则报告阻塞，不临时换库／算法。
- 参数接口参考[scikit-learn 1.5.2官方文档](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)。本协议参数是固定实验选择，不是文档推荐的最优参数。

## 2．拟合和阈值落锁

1. 先精确复现Phase92的输入及旧基线统计，再拟合G6／G7。核验`classes_=[0,1]`，Shadow概率取类1。
2. 预测概率统一转float32。每模型用全部153个fit代表的有效非Shadow像元，按原保守exact-ties规则冻结宏FPR≤1%的阈值。
3. 阈值保存为**概率域**，最终直接按`概率>=阈值`比较；不能再sigmoid一次。fit校准仍使用已训练样本，不宣称独立误报保证。
4. 两模型、阈值、输入／代码／依赖SHA全部落锁后，才评价development及ALCD。数值或资源失败报告失败，不缩样本、改参数或补第三个模型。

## 3．固定评价与报告

六行：Phase92的S3、R6、Source_spectral、L3，加G6、G7。

- development仍为32代表、7个有Shadow组；ALCD仍为9组原版全部支持及6组原／修订版共同支持，不改mask或成员。
- ALCD先逐20m像元预测树模型概率，再按Phase89的3×3均值形成60m分数，使用本模型冻结概率阈值。不能先平均波段再过树。
- 主要`G7−G6`宏AP；次要`Source_spectral−G6`、`G6−R6`。逐域列全部组差、±1百分点改善／稳定／下降、最差损伤；配对组bootstrap10000次、seed93012，区间仅描述。
- 报宏／池化AP、冻结召回／精确率、FPR及单侧95%上界、Shadow IoU、零召回、Surface与Cloud分层FPR和成本。只比较二分类，不给无Surface／Cloud读出的G6／G7伪造三类mIoU。
- 补充Phase91式1%宏FPR统一阈值oracle召回，另表标明使用评价参考扫描，不部署。两版共同支持分开复算，不挑有利版本。

## 4．收口与交付

若G6缩小旧差距，明确承认线性基线偏弱；若G7仍改善G6，仅支持这一固定非线性学习器下Source分数的实用增量，不自动归因为预训练、空间上下文或某种物理机制。若无明确增量，降低网络分数必要性主张；区间跨零不是等效。

无论结果都不调树深、轮数、特征或阈值，不选部署赢家。ALCD已反复使用，本轮仍是探索，不是独立确认或方法创新证明。

交付`outputs/phase93/phase93_20261012/`：短README、两个模型及阈值锁、六行总表、逐组CSV、双版本／oracle结果、依赖锁、拟合及推理成本、模型序列化重载一致性和CPU独立指标复算。旧恢复缓存谱系限制、人工B未执行及原空字段继续保留。

代码及小结果按原流程push／pull；不读取旧sealed像元、不改标签／历史结论，不同步大缓存或work_dirs；沿用2026-10-20北京时间预算截止。本文件仅立项，尚未拟合。
