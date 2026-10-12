# Phase90：外部参考上的三个固定输入对照

## 做什么

Phase89支持L3相对L2的外部参考AP增量，但不支持1%误报预算迁移。本轮复用Phase67已经冻结的三个对照，检查这项增量需要哪些输入。

**零训练、零拟合、零调阈值、零新增网络前向、零下载。**ALCD已经查看过，本轮是事后探索对照，不是新独立确认，也不重启LoRA分枝。

## 1．锁定输入与模型

- 固定Phase89的9组代表、原版主分析、6组双版本共同支持及全部有效mask，不换组、不补样。
- 使用已完成的`outputs/phase89/phase89_20261011/recovered_results/`报告与运行索引指向的恢复缓存。顶层旧README是故障历史，不能当成最终结果；恢复复用块的故障前SHA限制继续披露。
- 复用Phase67的`outputs/phase67/phase67_20261010/delivery/control_lock.json`。核验SHA、标准化、系数、clip、阈值与输入次序，禁止重新拟合。
- 先从缓存精确复现Phase89四策略的逐组分数／混淆矩阵，再从Catalogue既有缓存复现Phase67三个对照。异常停止排查，不修改旧锁。

| 对照 | 输入顺序 | 非Shadow读出 |
|---|---|---|
| L2_RGB | MsRE logit、Source logit、B04、B03、B02 | 原MsRE |
| Source_spectral | Source logit、B08、B11、B12 | 原Source |
| MsRE_spectral | MsRE logit、B08、B11、B12 | 原MsRE |

## 2．直接计算三个外部对照

1. 读取Phase89原Source／MsRE的**20m三类概率块**及20m六波段TOA；不能只用已平均的60m概率重算浅层分数。RGB使用TOA，不乘255。
2. Phase89波段存储顺序为B02、B03、B04、B08、B11、B12；L2_RGB新增三列必须按B04、B03、B02重排。网络分数clip及logit运算严格复用Phase67。
3. 在20m按各自冻结浅层模型计算logit，再sigmoid、3×3平均到60m；与`sigmoid(该模型原阈值)`按原>=比较。非Shadow类别使用表中对应网络的60m平均Surface／Cloud概率胜者。
4. 空间支持、边缘处理、精度、AP并列分数处理与Phase89完全一致。Source_spectral不得偷偷使用MsRE非云影读出。
5. 缓存缺失／校验失败时记录阻塞，不擅自补前向；不覆盖恢复缓存或旧结果。

## 3．固定比较和报告

七行总表：Phase89的Source、MsRE、L2、L3，以及三个新增对照。

- **主要：L3−L2_RGB。**同为五输入，检查额外NIR／SWIR相对额外RGB的效果。
- **次要：L3−Source_spectral。**检查在已有Source＋光谱之外，增加当前MsRE分数的效果。
- **次要：L3−MsRE_spectral。**检查Source分数在已有MsRE＋光谱之外的效果。

以上均重新拟合过各自系数的既有完整模型对照，不是单一系数的纯因果效应。AP／Shadow指标不受非Shadow读出影响，但三类mIoU差包含读出差异。

每个比较列9组原版全部支持的AP差、±1个百分点改善／稳定／下降、最差AP与召回损伤；配对相关组bootstrap10000次、seed90012，区间仅描述。三个比较不称正式多重检验通过。

七行同时列宏／池化AP、冻结召回、FPR及单侧95%上界、宏／池化mIoU、三类IoU、漏影流向、零召回、逐组支持和混淆矩阵。6组共同支持上分别用原版、修订版完整复算；同一版本间的策略使用同一mask。

## 4．怎么收口

- L3相对RGB对照若无明确优势，降低“额外光谱特异性”的主张。
- L3相对Source_spectral若无明确优势，不主张当前适配分数具有必要收益；相对MsRE_spectral同理，不主张Source分数必需。
- 区间跨零表示证据不确定，不能写成等效；两版方向不一致就明确披露，不挑有利参考版本。
- 无论结果不选新赢家部署、不在ALCD重校准阈值。若更简单对照值得继续，下一协议另定，不能把本轮结果回填成Phase89预先确定的主要比较。

交付`outputs/phase90/phase90_20261012/`：短README、七行总表、逐组CSV／混淆矩阵、三个配对差表、双版本共同支持表、参数／输入SHA锁和CPU独立复核。原人工意见及空字段保留，人工B未执行。

代码及小结果按原流程push／pull；不读取旧sealed像元、不改参考或候选、不自动同步大缓存或work_dirs。沿用2026-10-20北京时间预算截止。本文件仅立项，尚未执行。
