# Phase310：未接纳Surface区域的Source预测保持对照

## 一、做什么

在Phase304 Q90-PL上只增加一项约束：对冻结Source预测Surface、但未被Q90接纳的像元，保持Source同视野的三类软预测。检验这一约束能否减少Surface→Cloud损伤，同时保留Cloud→Shadow纠正。

本轮策略名 **Q90-SP**。一次全新LoRA训练、固定末步评价；与既有Source和Phase304 Q90-PL比较。不新增seed、阈值搜索、多光谱输入、EMA或监督探针。训练不使用目标参考、snow/ice场景标签、D/K事件或历史损伤排序。它是通用预测保持的机制对照，不预先称云特有创新。

## 二、怎么做

### 1．锁定Phase304基线，复用原资产

- 先读 `src/research_plans/phase304.md` 及 `src/research_results/phase304_20261024/` 实际执行锁、报告与训练日志；实现以 `src/tools/worker_phase304.py` 和其调用的Phase302路径为准。遇到计划/实际差异先记录并明确沿用实际身份，不静默替换。
- Source：`src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth`，SHA256 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。DINOv2-S、原Cloud-Adapter、neck/head/buffer冻结eval。全新LoRA rank4/alpha4/dropout0，全部12块qkv/proj/fc1/fc2，294912可训练参数。
- 原100 fit代表、32 development代表、1000步裁剪/翻转表、4固定探针不变。初始化与训练seed30201；逐张量核对Phase304 initial_lora。不得加载Phase304末步LoRA续训。
- 原RGB读取、影像valid、512裁剪、FP32、batch1和AdamW不变：lr1e-4、wd0.01、betas=(0.9,0.999)、eps1e-8、恒定学习率。不访问53校准、PixBox或其他评价池。
- 复用Phase304已保留的100产品完整Source s和Q90伪标签；先核验SHA、原始输入、scores定义/窗口及接受集合。不得使用Phase306样本包内的参考共同支持来定义训练valid。训练入口不能打开masks.zip、alt_masks.zip或classification_tags.csv。
- 原Q90定义：全图9窗等权拼接语义得分s后，q=s/sum(s)，valid且sum(s)>1e-12且max(q)>=0.90时接纳argmax(s)，其余ignore255。1000步预期955次非空、45跳过；接受总数75469158，三类[40622307,33554491,1292360]。本轮全部精确复现，不再重建100产品教师得分。

### 2．固定保护集合和损失，禁止事后改动

- 训练保护集合P完全由冻结Source全图缓存定义：`image_valid & (sum(s)>1e-12) & (argmax(s)==Surface) & (max(q)<0.90)`。由于Source预测Surface，等价于qSurface<0.90；不设新的置信下界、不扩张到高置信G90、不只挑雪冰/最大组、不依赖学生当前类别。
- P与Q90接纳集合A严格不交。它不等于Phase306的参考有效R，也不等于真值定义的D/K；训练中只有影像valid。逐产品及逐步保存P/A支持及掩膜SHA，二者按原裁剪/翻转同步取样。
- **软目标使用同一512裁剪视野的冻结Source**：每个原Q90非空步骤，对完全相同预处理/裁剪/翻转的输入，以无LoRA Source在eval/no_grad下求三类语义得分sT。另由学生求sS，二者空间网格相同；teacher不能读取学生LoRA、更新buffer或参与梯度。
- 固定温度T=1，`pT=softmax(sT,dim=class)`、`pS=softmax(sS,dim=class)`；约束 `L_SP=mean_{i in P} sum_c pT_ic*(log(pT_ic)-log(pS_ic))`。实际用log_softmax稳定计算，teacher detach，不加平滑/置信重权或类别权重。P为空时返回可反传的0，不产生NaN。
- 原损失 `L_PL=mean_{i in A} CE(sS_i,pseudo_i)` 不变。总损失 **L=L_PL+1.0*L_SP**；两项各自按像元数平均，不按A+P合并归一化。lambda=1固定，不扫描或根据结果调整。
- q只定义全图接受/保护掩膜；蒸馏使用softmax(s)，不偷换成q或CE(log(q))。这里保持的是既有得分尺度下的软分布，不称其为校准概率。
- 仅在原A非空的955步骤更新。45个A为空步骤仍跳过全部神经前向/优化，即使P非空也不单独训练；不补步。逐步记录A三类数、P数、两项原始/加权loss、总loss、梯度范数和更新标志。

### 3．训练前验收与一次训练

- 零LoRA与Source在相同512视野输出误差<=1e-6，初始L_SP应接近0（绝对值<=1e-6）；不得用全图拼接teacher软目标替代同512 teacher，使零LoRA先天承受视野差异。
- 验收P/A交集为空；invalid不进入两项loss；P为空、A为空的分支正确。KL方向用不同三类分布手算小例复核，不把交叉熵误记为KL。Source参数无梯度，冻结head允许梯度传向LoRA。
- 在原裁剪表中取首个A非空步骤进行一次lambda=0验收：同一初始权重/RNG下，与Phase304实现的L_PL、LoRA梯度和一步更新逐张量比较，最大绝对差<=1e-6；不因差异改变输入、容差或历史文件。其余门检可选P非空的首个原步骤，不人工挑景。
- 所有门检结束后恢复初始LoRA、优化器、梯度与RNG，再正式训练。记录各步骤teacher/student实际调用，门检亦计预算。
- 第0/100/500/1000尝试步在原4中心512探针保存学生s/pred/valid，共16次；不据探针挑模型或早停。初末基础权重和buffer逐张量一致，优化器仅含LoRA。
- 原1000尝试步末权重、训练日志、掩膜/代码锁封存后，才准入本轮目标参考分析；历史development已暴露必须披露，不能声称研究全程未接触标签。

### 4．固定末步评价与损伤保留分析

- 评价原32组，每产品512窗、步长256、起点[0,256,510]²，9窗语义得分等权拼接；只新增Q90-SP的288次前向。Shadow评分沿用softmax(s)[2]，不用q[2]替换。缺类规则与共同参考支持沿用Phase304，不删除单类组。
- Source和Q90-PL预测复用已核验缓存。正式评价前，用Phase304原末步权重对按产品ID排序的首个development产品重放9窗，与其缓存s误差<=1e-6、pred完全一致；这是旧模型身份门检，不重新选择对照。如果失败先查环境/身份，保留失败，不自动全量重跑基线或拿不同协议比较。
- 同报三类IoU、组均/池化mIoU、Shadow宏/池化AP、宏召回、宏/最大组误报。主比较Q90-SP−Q90-PL，另列两者各自相对Source差值；完整32逐组数据及union类数另交。
- 对Source→Q90-SP及Q90-PL→Q90-SP均构建完整27格事件张量并验证混淆边缘。全部六种有向错误列旧数、新数、消除、新增和转成其他错误，不只报净数。
- **损伤主要终点**：在原Source正确Surface像元中，Q90-PL新增Cloud的原506798个D里，Q90-SP修复为Surface多少、仍Cloud多少、转Shadow多少；同时列原D外Source正确Surface新增Cloud数，防止修旧损伤却在别处制造新损伤。Phase306非G D=450058仅作已知口径复现，不能替代训练P或总损伤分母。
- **收益保留终点**：在truth=Cloud、Source=Shadow、Q90-PL=Cloud的原784478个纠正像元中，Q90-SP仍Cloud多少、退回Shadow多少、转Surface多少；同时列该集合外新增Cloud纠正。不以Cloud→Shadow错误净减少879124冒充实际纠正数。
- 在参考有效支持下，固定按训练定义P、Q90接受A、其余区域分层上述损伤与纠正；dev保护集合也由Source缓存纯影像定义。P内并非都是真Surface，另列P内真Cloud/Shadow误保护代价，不要求它们为零。
- 全32组保留；最大T25XEG及Phase306固定前三组仅附诊断，不按结果改选择。snow/ice等场景标签只在封存后附表统计，注明场景标签不证明像元真值。Phase309已解释窗口尺寸差异，不再当错位证据；逐景仿射与雪冰参考歧义继续披露。

## 三、交付什么

输出 `outputs/phase310/phase310_YYYYMMDD/`，摘要 `src/research_results/phase310_YYYYMMDD/`，日期取实际开始北京时间。RESULT.md两页内，分“数据结果、结论分析”；完整CSV另交。

**表一：训练身份与新增约束支持（TRAINING_SUPPORT.csv）。**

| 策略 | 初始权重/日程SHA | 尝试/更新/跳过 | A三类数/出现步 | P唯一像元/累计裁剪像元/非空步 | PL/SP损失摘要 | 基础权重及buffer不变 |
|---|---|---|---|---|---|---|

**表二：固定32组评价（METRICS.csv）。** Source、Q90-PL、Q90-SP及差值全部列出；逐组明细不删NA。

| 策略 | 全部/AP有效组数 | 组均/池化mIoU | 三类IoU | Shadow宏/池化AP | 宏召回 | 宏/最大组误报 |
|---|---|---|---|---|---|---|

**表三：损伤修复与收益保留（DAMAGE_BENEFIT.csv）。** 全池、P/A/其余区域及32逐组均交，零分母记NA。

| 区域/组 | 原D支持 | D转Surface/Cloud/Shadow | 原D外新增Surface→Cloud | 原Cloud纠正支持 | 保留Cloud/退回Shadow/转Surface | 集合外新增纠正 |
|---|---|---|---|---|---|---|

**表四：六类错误流与保护代价（ERROR_FLOW.csv）。** 两种比较、三种区域及所有真值类完整交付，27格另CSV。

| 比较/区域 | 真值支持 | 错误方向 | 旧/新数 | 消除/新增/净变化 | 消除中改正/另一错误 | 错误率差pp |
|---|---|---|---|---|---|---|

**表五：同512探针（PROBES.csv）。** 4探针×全部固定时点×三类及P/A/其余分层；不选模。

| 产品/时点/区域 | 真值类及支持 | Source正确→本轮错 | Source错→本轮正确 | Q90正确→本轮错 | Q90错→本轮正确 | 相同视野核对 |
|---|---|---|---|---|---|---|

另交初末LoRA、训练日志与曲线、teacher软目标定义及实现、保护/接受集合锁、冻结/梯度/KL/lambda0门检、评价缓存、CPU指标与事件复算、代码/输入/参数/SHA清单、完整资源/失败/备份校验记录。teacher同512输出不必整批持久化，仅保存门检样例与逐步支持/调用证据。

报告直接回答：损伤是否减少、是否转移到其他像元/错误类、Cloud纠正保留多少、最大组是否仍失败；两项loss及梯度是否有效。一次固定lambda失败不等于所有保持约束无效，成功也不能证明损伤完全源于未接纳区域。它与Phase304改变的仅为新增约束及其teacher计算；精确重放门检通过仍不是跨seed稳定性证据。不得根据结果追加lambda/seed、改变参考或开启下一phase。

预算：正式训练955次学生+955次teacher，评价288、探针16、旧基线重放9，共2223次前向；所有门检/失败/恢复计入新增总数<=2400。GPU阶段<=4小时、单GPU任务，启动先查冲突；沿用gzs cloud-lite-pt210及原算子环境，不升级/下载。CPU<=16GiB，新持久产物含远端与本地备份<=4GB，宿主空闲内存及磁盘均>=4GB；原缓存只读引用不复制。缺关键缓存或预算不足先交缺项，不降分辨率/删组/换精度补齐。计算在gzs，本机仅编辑/备份/哈希。本文件仅立项，不启动远端执行，不新增推送/定时检查授权，不自动开始下一phase。

## 执行授权补充

用户明确要求“执行310”，沿用本聊天本轮代码、实验结果push/pull及定时检查授权。唯一一次训练前须通过全部门检；计算仅gzs cloud-lite-pt210，本机仅编辑、文件SHA及完整备份。完成后删除本轮检查，不启动下一phase。
