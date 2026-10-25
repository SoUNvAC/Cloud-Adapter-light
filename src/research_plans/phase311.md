# Phase311：单向Cloud−Surface得分差约束

执行授权补充（2026-10-25）：用户已明确下达“执行311”，沿用本聊天的代码/结果push、gzs pull及定时检查授权。所有模型、CPU分析和曲线均在gzs，本机仅编辑、文件SHA和完整备份。历史缓存只读。为避免Phase310流式解码故障，311先生成一次有预算的归档，传回后整包SHA先验，再逐文件验SHA；远端和本地归档均计入4GB上限。失败保留，不盲重试。

## 一、做什么

沿用Phase310的全部训练设置和保护集合，只把三类KL替换为单向Cloud−Surface得分差约束。检验减少Surface→Cloud损伤时，能否比Phase310保留更多Cloud纠正、减少Shadow误报。

策略名 **Q90-CM**。一次全新LoRA训练；固定对照Source、Phase304 Q90-PL、Phase310 Q90-SP。不扫权重、不加seed，不增加多光谱输入或监督探针。本轮是通用约束形式的机制修订，不能据此宣称云特有方法已成立。

## 二、怎么做

### 1．固定身份、输入和保护范围

- 先读 `src/research_plans/phase310.md`、Phase310实际执行锁及 `src/research_results/phase310_20261025/RESULT.md`。模型、输入、日程、伪标签、优化器、冻结状态和评价支持均以Phase310已验收的实际版本为准；本文件下述新损失是唯一训练策略改动。历史“执行310”的推送/运行授权不自动成为“执行311”的授权。
- 原Source：`src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth`，SHA256 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。DINOv2-S和原Cloud-Adapter/neck/head/buffer全部冻结eval；LoRA rank4/alpha4/dropout0、12块qkv/proj/fc1/fc2，294912参数。
- 从原Source全新初始化LoRA，seed30201，与Phase304/310初始LoRA逐张量一致。不得续训任一历史末步权重。沿用原100 fit代表、32 development代表、1000步512裁剪/翻转表、4探针及FP32/batch1。
- AdamW仅含LoRA：lr1e-4、wd0.01、betas=(0.9,0.999)、eps1e-8，恒定学习率。原RGB读取/预处理/影像valid不变；不使用参考ignore定义训练支持。训练进程不得读取masks、alt_masks、场景标签或D/K样本包。
- 从Phase304/310原缓存按SHA复用Q90集合A：valid且sum(s)>1e-12且max(q)>=0.90，其中q=s/sum(s)，伪标签argmax(s)。原100产品三类接受数[40622307,33554491,1292360]，合计75469158，原955步更新/45跳过必须复现。
- 保护集合P仍为 `image_valid & sum(s)>1e-12 & argmax(s)==Surface & max(q)<0.90`，由Source全图缓存生成；与A不交。P唯一产品像元8201741、955更新裁剪累计19937970；按Phase310逐产品/逐步骤掩膜SHA和支持复现。该P不等于参考有效非G区域或真Surface，也不随学生变化。

### 2．唯一修改：单向得分差损失

- 每个原A非空步骤，对完全相同512裁剪、翻转和预处理分别计算冻结Source的sT和学生的sS。Source独立、eval/no_grad，无LoRA；三类s仍为Mask2Former排除no-object后的语义得分，不是原生logit。不要改成q、softmax(s)或全图拼接teacher目标。
- 定义 `mT=sT_Cloud-sT_Surface`，`mS=sS_Cloud-sS_Surface`；teacher detach。定义 `d=relu(mS-mT)`，固定约束 **L_CM=mean_{i in P}(d_i^2)**。均值分母为全部P像元数，不是d>0的像元数；P为空返回可反传的0。
- 总损失固定为 **L=L_PL+1.0*L_CM**，其中 `L_PL=mean_{i in A} CE(sS_i,pseudo_i)` 完全沿用Phase304。两项分别平均，无温度、margin常数、额外归一化、类别重权或自适应lambda。
- mS<=mT时不惩罚；mS>mT时压低学生Cloud相对Surface的得分差。本约束不要求Source预测Surface的像元被硬判Surface，且Source同512也可能与全图类别不同，不另筛掉这些像元。
- 损失不直接含sS_Shadow，因此对作为独立输出变量的Shadow偏导为0；但共享参数、Surface提升和argmax竞争仍可改变Shadow结果。不得宣称“不影响Shadow”或“已分离云影适配”。
- 原45个A为空步骤继续跳过全部神经前向/优化，即使P非空；不补步、不单独做CM更新。

### 3．训练前验收、正式训练和机制日志

- 零LoRA同512得分与Source最大误差<=1e-6，初始CM<=1e-10；冻结基础参数/buffer不变，梯度可穿冻结head进入LoRA。掩膜/分母/标签入口沿用Phase310验收。
- 用独立小张量验证：mS低于/等于/高于mT、P为空、部分invalid、teacher detach。用手算验证高于时对Cloud/Surface的梯度分别为+2d/|P|与−2d/|P|，Shadow为0；等于时ReLU梯度按实现取0。禁止用真实目标标签做门检选择。
- 在原日程首个A非空步骤做lambda=0验收：相同初始权重/RNG下与Phase304的PL、LoRA梯度和一步更新最大绝对差<=1e-6。门检结束恢复初始权重、优化器、梯度和RNG，再开始唯一一次正式训练。
- 正式1000尝试/955更新；逐步保存A三类数、P数、P内d>0数/比例、mT/mS均值、d均值/最大值、PL/CM/total、LoRA总梯度范数。P=0或跳过步骤的统计按明确NA/0规则记录。
- 在固定尝试步1/101/500/900记录PL与CM的LoRA梯度范数，复用本步计算图；不额外优化、不改变更新方向。初始CM梯度为0正常，后续若近零或约束很少激活照实报告，不自动放大lambda。平方得分差与KL的数值尺度不同，lambda同为1不代表约束强度匹配，不能把全部差异唯一归因于“是否约束Shadow”。
- 第0/100/500/1000尝试步沿用4探针，各保存s/pred/valid。末步权重、日志、代码/输入锁封存后才准入参考分析；不按探针选模、改lambda或补训。

### 4．固定末步评价和收益代价核算

- 原32 development、每产品9个512窗口、起点[0,256,510]²、语义得分等权拼接；新增学生评价288次。Shadow评分沿用softmax(s)[2]。全32组及原缺类规则保持不变，不访问其他评价池。
- 三个历史对照缓存需锁定权重、输入、支持、窗口/得分定义和SHA。对按产品ID排序首个development产品，分别用Q90-PL和Q90-SP原末步权重重放9窗；各自与缓存s误差<=1e-6、pred一致。共18次身份门检，失败查原因，不静默换协议或全量重跑对照。
- 全报Source/PL/SP/CM的组均及池化mIoU、三类IoU、Shadow宏/池化AP、宏召回、宏/最大误报；CM−PL、CM−SP分别列。Source只作固定基准，不改历史指标。
- 主要损伤口径固定：Source正确Surface、Q90-PL错判Cloud的原D=506798。分别列CM转Surface/仍Cloud/转Shadow，以及原D外Source正确Surface被CM新判Cloud、判Shadow的数量。不能只看“原损伤修复率”。
- 收益口径固定：truthCloud、SourceShadow、Q90-PLCloud的原纠正集合784478。列CM保留Cloud/退回Shadow/转Surface及集合外新增纠正。对照SP原保留362132、退回Shadow417682、转Surface4664；纠正数不能与Cloud→Shadow净错误变化混用。
- Source→CM、PL→CM、SP→CM均交完整27格及六类有向错误流，并核对混淆边缘。固定按P/A/其余区域分层，P由纯影像Source定义；所有真值类保留，明确保护区真Cloud/Shadow误保护代价。
- 最大T25XEG单列三类像元支持、IoU、union类数和Surface→Cloud/Shadow计数；区分全D206076与Phase308诊断子集197082，不混分母。Source全Surface结果及SP新增33087个Shadow误报是历史参照，不能因单类组计分敏感而删组。
- 全32逐组及Phase306固定前三组附表保留。参考边界/非边界沿原定义附表，不称物理薄云边缘；场景标签仅封存后分析，雪冰标签不当像元真值。Phase309未决的雪冰参考与逐景仿射限制继续披露，不重开溯源分枝。

## 三、交付什么

输出 `outputs/phase311/phase311_YYYYMMDD/`，摘要 `src/research_results/phase311_YYYYMMDD/`，日期用实际开始北京时间。RESULT.md两页内，分“数据结果、结论分析”，完整CSV另交。

**表一：支持与约束激活（TRAINING_SUPPORT.csv、CONSTRAINT_ACTIVITY.csv）。** 唯一像元与累计裁剪支持分开，完整1000步另交。

| 策略 | 初始/日程/掩膜SHA | 更新/跳过 | A三类/P支持 | P内激活数/比例 | PL/约束/总loss | PL/约束梯度范数 | 原权重/buffer不变 |
|---|---|---|---|---|---|---|---|

**表二：固定评价（METRICS.csv）。** 四策略及两组主要差值，逐组表含NA和union类数。

| 策略 | 全部/AP有效组数 | 组均/池化mIoU | 三类IoU | Shadow宏/池化AP | 宏召回 | 宏/最大误报 |
|---|---|---|---|---|---|---|

**表三：原损伤与纠正的去向（DAMAGE_BENEFIT.csv）。** PL/SP/CM同分母，交全池、P/A/其余和全32组。

| 策略/区域/组 | 原D | D转Surface/Cloud/Shadow | D外新增Surface→Cloud/Shadow | 原纠正集合 | 保留Cloud/退回Shadow/转Surface | 集合外新增纠正 |
|---|---|---|---|---|---|---|

**表四：完整错误流与误保护代价（ERROR_FLOW.csv、PROTECTION_COST.csv）。**

| 比较/区域 | 真值类支持 | 错误方向 | 旧/新数 | 消除/新增/净变化 | 消除中改正/转另一错误 | 错误率差pp |
|---|---|---|---|---|---|---|

**表五：同视野探针（PROBES.csv）。** 全4探针×4时点×3类，P/A/其余分层；历史策略只在已有匹配时点/视野/支持时准入，否则NA，不补推理。

| 产品/时点/区域 | 真值类/支持 | Source正确→CM错/Source错→CM正确 | PL正确→CM错/PL错→CM正确 | SP正确→CM错/SP错→CM正确 | 上下文身份 |
|---|---|---|---|---|---|

另交初末LoRA、完整训练日志/曲线、原缓存引用锁、门检小例与模型验收、评价s/pred/评分/valid、CPU指标与27格复算、代码/输入/环境/SHA锁、资源/失败/完整备份校验。teacher逐步得分不用整批持久化，保留门检样例和调用记录。

报告直接回答：相对PL是否改善取舍；相对SP是否保留更多Cloud纠正且减少Shadow误报；约束是否实际激活；修复是否转移成别处损伤；最大组是否仍失败。若只改善一个维度，按实际代价解释；若仍失败且无新证据，不继续同一保护区上无上限换loss/扫lambda，本分枝记录到此，下一步另提云特有假设。当前结果无论方向如何，都不证明跨seed稳定性或独立目标域收益。

预算：955学生+955teacher+288评价+16探针+18历史重放=2232次计划前向；门检/失败/恢复全部计入新增总数<=2400。GPU阶段<=4小时、单GPU任务，启动先查冲突；沿用gzs cloud-lite-pt210和原算子环境，零下载/升级。CPU<=16GiB、新持久产物含远端与本地完整备份<=4GB、宿主空闲内存及磁盘均>=4GB。原缓存只读引用，缺失先报告具体缺项，不降分辨率/删组/换精度。计算全部在gzs，本机仅编辑、文件SHA与备份。本文件只立项，不启动执行或新定时检查、不自动push或开始下一phase。
