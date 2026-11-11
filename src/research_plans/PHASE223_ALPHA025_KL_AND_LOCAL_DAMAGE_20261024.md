# Phase223：低强度适配下的教师保持与局部云影损伤

## 1．执行内容

零训练、反传、拟合、调参。固定α=0.25，评价以下8个学生head：

| 目标head来源 | seed | 臂 |
|---|---|---|
| Phase217 SPARCS | 21701、21702 | C、D |
| Phase222 L8 | 21701、21702 | C、D |

只运行原535张源val／107 ROI。原Source为共同教师。比较同α的D／C最终教师KL、参考错误与直接转移，重点核对ROI_0124，但全107 ROI全部交付。

SPARCS seed21701按联合规则实际选α=0.5，本轮统一0.25是适配强度匹配诊断，**不能将其称为两个目标域全部已选工作点的诊断**。不补α=0.5或1，不运行目标集、Replay256、官方源test、L8新输入或Omni。本聊天只立项，未启动远程资源。

## 2．模型与支持锁

- Source SHA：`64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。
- 四个SPARCS末步head按217、四个L8末步head按222训练完成回执SHA恢复；按原插值代码构建α=0.25，核对旧插值head SHA。不能改用最佳checkpoint、历史seed64或其他训练臂。缺原件报告，不重训。
- 全部eval、FP32，冻结教师、完整backbone和所有学生参数／buffer。精确沿用217／222最终验收的评价设置、RGB512、Source父类映射、ignore和输入清单。
- 同一输入共享一次冻结backbone特征、一次教师head、8次学生head。每个正式输出必须与原正式路径一致，不通过改语义归一化或推理路径获得“近似复现”。
- 源val参考仅用于诊断，不反馈梯度，不生成新回放清单或训练掩膜；train标签和官方test禁止读取。既有材料只读。

## 3．KL与转移口径

唯一KL沿用218：

```text
S = einsum(last_query_class.softmax(-1)[..., :-1],
           sigmoid(bilinear(last_query_mask, size=512, align_corners=False)))
logp = log_softmax(S_Source.float(), class, T=1)
logq = log_softmax(S_student.float(), class, T=1)
KL_i = sum_class exp(logp_i) * (logp_i - logq_i)
```

教师／学生argmax照原正式预测路径；semantic-softmax置信度不称校准正确概率。KL按共同几何支持汇总，参考ignore另列；负舍入值原样记录，不靠clamp掩盖。

每图／目标／seed直接保存整数`J[参考,教师,C,D]`，shape=(3,3,3,3)。另保存按参考类、教师／学生正确性、六种错误流分组的KL和／像元数，以及教师—学生argmax分歧数。不从旧混淆反推同像元转移。

参考Shadow且教师预测Shadow的区域，分别统计C／D判成Surface／Cloud的像元数、平均KL、教师置信度；对“教师正确Shadow→学生Surface”再报teacher与student的原semantic有符号差`S_Shadow-S_Surface`的中位数／P10／P90及支持。这是输出诊断，不把分数差当训练因果解释；空组NA，不能填0。

## 4．最终提交三张表

### A．全体源val

| 目标 | seed | C/D | 平均KL | 教师分歧率 | 源池化损失 | T11 | 最差ROI及损失 | 错误像元 |
|---|---|---|---|---|---|---|---|---|

列D−C的KL／错误计数差及各参考类分解。列出全107 ROI中“D的KL更低但错误更多”的数量及清单，不只展示ROI_0124。源损失和T11复现原同α结果，不新算置信区间。

### B．固定焦点ROI_0124

| 目标 | seed | C/D mIoU | C/D KL | C/D参考Shadow KL | C/D Shadow召回 | Source/C/D Shadow→Surface计数 |
|---|---|---|---|---|---|---|

附教师原正确Shadow上的新增漏影、C错→D对／C对→D错、上述有符号差诊断。原最差11 ROI及全部107 ROI同格式交CSV，不能从ROI编号猜地物。总mIoU分解与类错误计数分别解释。

### C．预测保持与参考正确性

| 目标 | seed | ROI／全体 | 教师原正确区域：D比C少错像元 | 教师原错误区域：D比C少纠正像元 | D−C总错误 |
|---|---|---|---|---|---|

验明：总错误变化＝少纠正像元−少错像元。此表来自本轮同像元输出，但表中两个差仍是固定教师区域内的净计数；另交J导出的直接C→D转移，不混淆两者。

## 5．复现、预算和交付

固定源val第一图作工程门检：克隆Source的KL近0、共享与原正式路径一致、教师和8学生无梯度且参数／buffer不变；复用正式输出不重复算样本。门检后逐图、逐ROI及池化混淆精确复现217／222的同α缓存，T11和ROI_0124数值复现本地审计`outputs/research_audits/cross_target_residual_damage_20261024/`。旧预测SHA若存在同时核对；不能仅凭mIoU接近通过。

预计535 backbone＋535教师head＋4280学生head＝5350单位。总新增单位≤6000、GPU阶段≤1小时、GPU≤20GiB、CPU≤16GiB、新持久文件≤2GB、宿主物理盘余量≥4GB。失败／门检／恢复全部入账，超限先报缺口，不删组、学生或不利输出。零下载、环境升级、自动push和定时任务，无日历截止。

独立CPU验收用另一套显式轴求和重建J的参考支持、Source/C/D混淆、直接转移、正确性恒等式；固定少量像元保存teacher/student的semantic或logp/logq，逐类显式复算KL，允许事先明确的FP32舍入误差1e-6。大幅负KL、预测漂移或输入错配先定位，不事后放宽容差。

输出`outputs/phase223/phase223_20261024/`：两页内`RESULT.md`、表A/B/C、完整逐图／ROI／类分组KL、联合计数及轴定义、teacher/8学生预测SHA、输入／head／代码锁、冻结／原结果复现回执、独立验收和真实成本。无需完整logits／特征／影像副本。

结论只判断：在同α下D是否同时改善平均教师保持和平均参考表现；ROI_0124是否仍有KL下降伴参考损伤、或其教师漂移仍明显；教师原正确Shadow的边界变化是否与漏影对应。**本轮不测训练回放覆盖、时间轨迹或梯度因果，不能据此证明“训练没有看够该类”、命名新机制或自动启动新训练。** 两目标域共用同107源ROI，重复观察不增加地理独立样本量。
