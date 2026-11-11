# Phase212：VFM解码头权重插值保源探针

## 1．执行范围

复用Phase64已训练的Source及L8／SPARCS两个shared-parent目标解码头；冻结完整DINOv2／Cloud-Adapter backbone。每个目标头只新增三个固定权重插值点：**α=0.25、0.5、0.75**，同时评价该目标域val及CloudSEN-val。

**零训练、零拟合、不选新checkpoint、不调阈值、不新增seed。** 两个目标域都执行，不按第一域结果决定是否运行另一域。不新增门控、正则、输出集成或其他训练臂。用户已批准本轮立项，执行者可在下述身份与预算检查通过后完成固定探针，无需重复确认。

本轮同时服务两项探索：①简单保源基线能否改善已有目标／源能力取舍；②固定VFM体系的解码头适配价值。它不完成标签量曲线，也不预设插值是一项新方法。

## 2．先找原件并核验身份

优先读取现有远端或已封存归档，不下载新权重／影像、不重训补缺。仓库相对路径如下，远端按原仓库位置解析：

| 模型 | 原路径 | 预期SHA256 |
|---|---|---|
| 三父类Source | `src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth` | `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9` |
| L8 shared-parent | `src/work_dirs/phase64_target_parent/l8/shared_parent/seed64/best_mIoU_iter_3500.pth` | `2d9dfac9a8182445e52cc284cbd475f3c1ed413873305e2215565e50d3c576fe` |
| SPARCS shared-parent | `src/work_dirs/phase64_target_parent/sparcs/shared_parent/seed64/best_mIoU_iter_3000.pth` | `69a49f09718a7e0a132a05ea36fb8d9150f90087233a3b1b3475a86c43b38827` |

来源：`src/research_plans/PHASE64_FINAL_RESULT_AUDIT.md`及`src/work_dirs/phase64_pareto/summary.json`。后者是旧摘要，**只复用目标头身份等未被修正的字段；不得照搬其中旧四类Source基线和旧遗忘结论。**

- 需重建模型时，沿用Phase64原DINO初始化、转换及runtime，先核验实际文件与加载清单。稀疏checkpoint须按原方式恢复完整模型，不能把缺失backbone当随机初始化。
- 两目标模型的完整backbone参数／buffer须与三父类Source逐张量相同；decode_head结构、参数名、形状和类别顺序一致，才允许插值。若存在neck等其他组件，确认同样未变化。
- 原件缺失、SHA不符、完整模型无法恢复或冻结状态不符，交一次明确缺项与可恢复路径；不得改用同名其他checkpoint或自动重新训练。

## 3．只实施这一种插值

对每个目标域分别构造：

`head_alpha = head_source + alpha * (head_target - head_source)`

插值对象是**完整decode_head的全部可学习浮点参数**，含pixel decoder、query decoder、query embeddings及分类／mask投影等；不是只插最后分类层。backbone／Cloud-Adapter及其他组件使用原Source，完全冻结、eval，零反传和优化器更新。

- CPU float64计算插值后转回原参数dtype；原推理为FP32。α=0／1直接加载原端点，不用运算重构端点。
- 先审计全部buffer。原两端buffer预期相同，相同者原样保留；非浮点状态不参与插值。若buffer不同，保留差异并停止该域准入，不能临时引入BN重估或混合策略改变端点。
- 固定三个α在查看新性能前锁定。逐参数核验公式、有限性、形状及backbone不变，保存三个新head的SHA。
- 不平均两端输出代替权重插值。Mask2Former是非线性头，两者不是相同操作。各域的插值头都同时在源域及本域开启评价，不按域切回Source规避损伤。

## 4．固定评价支持与端点复现

只使用Phase64原清单与其最终审计口径：

- 源域：CloudSEN-val原535张，映射到Surface／Cloud／Shadow三父类。
- L8：原`Shadows?=yes` target-val，最终审计的963个patch／8个scene；原target-train1560密集patch／18scene只作预算背景，本轮不读训练标签。
- SPARCS：原scene-disjoint target-val原清单，不把全75scene或原生scene数当实际模型输入次数。按manifest列明实际输入数、独立scene数及标签映射；不扩到test。原60个密集训练scene只作预算背景。

全部输入缩放、RGB数值处理、类别映射、ignore及原模型预测读出沿用Phase64实际评价代码，优先核对`src/tools/evaluate_phase64_rgb_recheck.py`及原训练评价日志。不要引入Omni的B8A、归一化或后处理。

先复现α=0、α=1：原Source在源域只跑一次，可供两个域共用；其他端点在各自源／目标支持上复用合格预测缓存，缺缓存才补前向。α=1核对原target最佳结果及源域开启结果。历史表只到两位小数时，按原精度舍入匹配；有精确混淆或预测SHA时必须精确核对。发现评价器差异，定位并记录，不改终点数值凑表。

最终参考值（%）：

| 目标域 | Source目标mIoU | 目标头α=1目标mIoU | Source源mIoU | 目标头α=1源mIoU |
|---|---:|---:|---:|---:|
| L8 | 58.0530 | 64.10 | 77.8179 | 63.21 |
| SPARCS | 55.5775 | 69.65 | 77.8179 | 69.92 |

历史MsRE seed64仅作参照：L8目标61.0182／源73.3432；SPARCS目标67.1705／源73.7240。LoRA为L8目标62.80／源66.6042，SPARCS目标68.82／源68.1511。不重跑它们，不引用原“关闭恢复77.81”冒充开启结果；不拿MsRE三seed均值与本轮单seed作配对胜负。

## 5．指标与结论规则

**主要结果是每个α的“目标池化三类mIoU、源池化三类mIoU”二元坐标。** 两者都越高越好。另报两域各类IoU、逐组平均mIoU、Shadow召回、非Shadow→Shadow误报率、最大组误报及完整混淆矩阵。误报分母为有效参考Surface＋Cloud；无影组召回NA，误报保留。

保留所有α，不选一个赢家部署。不新增α扫描、早停或新阈值。目标头此前已按target-val选中，本轮源val也会用于探索，不能将结果称为独立确认或无偏测试。

固定描述比较：

1. α=0.25／0.5／0.75分别相对Source和目标头端点，目标收益与源损失是多少。
2. 各内部点与已有MsRE／LoRA坐标比较。只有目标和源mIoU同时不低、至少一项更高，才称该点**点估计支配**该历史坐标；必须用未舍入数值，来源精度不足时标“待精确值核验”，不能用近似小数断言支配。
3. 只在本轮固定点和既有方法坐标组成的有限集合内标非支配点，不称理论最优前沿，不称普遍战胜某方法。单纯处于两个端点之间不等于新方法推开前沿；超出图上两端点连线也不是机制证明。

不设新的“涨1点／区间下界过零”晋级门，不强求保源与目标各指标全部提高。曲线若只显示普通折中也如实收尾。

对每个内部α相对两个端点的池化mIoU差，做10000次配对整组bootstrap，seed212020；每次汇总抽中组混淆矩阵后重算mIoU。源域用既有已核验相关组／ROI分组，目标用原scene；不能把535张源patch当535独立scene。源分组身份不够时只报告源点估计及分组缺口，不临时按性能分组。区间条件于现有checkpoint，不包括训练／checkpoint选择不确定性；不能用两个独立区间重叠与否代替二维前沿检验。

## 6．预算与执行注意

先按现有清单锁定调用预计值，三内部点都须评价完整源／目标支持。预计量为`3×[(535+L8输入数)+(535+SPARCS输入数)]`，另加缺失端点及门检；缓存可用则扣减实际调用，不将scene数代入输入数。

总新增模型调用≤16000次、累计GPU阶段≤4小时，包含端点、失败和恢复；GPU≤20GiB、CPU≤16GiB、新持久文件≤4GB、宿主物理盘余量≥4GB。只保存必要head、逐输入／逐组混淆、账本及小型验收证据，不保存全量中间特征／四类logits。元数据预计超预算或工程失败无法在余量内恢复时报告现状，不删场景、改精度或缩图凑预算。

零下载、零环境升级、零训练。禁止读取CloudSEN test、L8／SPARCS target-test、新确认池；不启动定时任务、不自动push，不自动进入下一phase，无日历截止。

## 7．交付给我

输出`outputs/phase212/phase212_20261020/`，摘要`src/research_results/phase212_20261020/`。`RESULT.md`两页内，数据与解释分开，主表为：

| 适配目标 | α／历史方法 | 目标池化mIoU | 源池化mIoU | 目标收益vs Source | 源损失vs Source | 目标Shadow IoU | 源Shadow IoU | 有限集合非支配状态 |
|---|---|---|---|---|---|---|---|---|

每域α=0／0.25／0.5／0.75／1五行，加原MsRE／LoRA历史参照。另交逐组完整指标、配对差及区间、参数／buffer／backbone验收、原件与新head SHA、评价清单／标签映射锁、端点复现、独立CPU混淆与bootstrap核验、调用／资源／失败账本。

分析部分只回答：最简单的头部插值能否获得有用保源折中；现有MsRE／LoRA是否仍保留未被简单基线覆盖的点；是否值得下一步针对性研究。不得据此宣称已做标签效率曲线、保住通用DINO全部能力、建立因果机制或形成新算法。WiSE-FT式插值作为已知基线披露，不能改名包装创新。

本聊天仅立项，未启动远程检查或推理。
