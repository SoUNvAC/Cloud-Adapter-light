# Phase317：用现有三份数据完成第一轮无标签基线对照

## 一、做什么

继续研究，但先回答一个简单问题：**普通自训练的云影损失，能不能用最简单的类别均衡训练缓解？这个现象在手头的另外两份数据上是否还出现？**

固定CMC、SPARCS、L8 Biome三个数据集，对照Source、Q90普通自训练PL、同接纳集合的类别均衡自训练CB。这不是新方法论文实验，也不宣称复现了原版CBST或SHOT。它为限时方法试验建立可比较的起点：若常用均衡方式已解决问题，就不能把相同做法包装成新方法；若不能，再根据实际失败提出一个有针对性的新假设。

把现有接口检查、必要实现、训练和结果整理放在本轮完成，不再拆成一串资料准备phase。新增训练最多5臂、每臂一个seed，不扫参数，不恢复旧SP/CM或光谱分枝。

## 二、怎么做

### 1．固定Source、数据来源与历史身份

- 原Source仍是 `src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth`，SHA256 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。它来自CloudSEN12 Source路线；不加载历史L8/SPARCS密集监督训练过的目标head、MsRE、LoRA或优化器。
- Source及原Cloud-Adapter/neck/head/buffer冻结eval；LoRA rank4/alpha4/dropout0、12块qkv/proj/fc1/fc2、294912参数。所有新增臂seed30201，从同一个原Source和同一新LoRA初始状态开始。独立保存初始A/B，核对与Phase304初始状态一致。
- 使用用户已实查的 `/home/scv/shared/data` 中现成文件，不新增下载、不重新清点138GB数据。只读加载所需的具体成员/manifest并核验必要SHA；目录存在不代表实际编码已验收。
- CMC、SPARCS、L8均是历史开发使用过的数据，**本轮是多数据集探索，不是新盲测**。旧SPARCS/L8监督结果只证明接口存在，不能算本轮SFDA收益。三个数据集不等于三个传感器，SPARCS与L8都来自Landsat-8。
- 参考 `src/research_plans/EXISTING_DATA_REASSESSMENT_20261026.md`。不要求先补齐未见确认池才开本轮，也不因历史暴露而隐瞒评价限制。

### 2．使用哪些图像与输入

**CMC：** 原100fit、32development、1022原TOA RGB、1000步裁剪/翻转表和9窗评价完全沿用Phase304。复用已核验Source scores、Q90掩膜及30201的PL端点，只新增一个CB训练臂。不得把新CMC日程改为全图缩成512，以免失去与既有PL的可比性。

**SPARCS：** 使用既有已验收的scene manifest，按 `src/configs/protocol/phase64_target_parent_rgb.py` 定位 `phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv`。预期旧train60、val10，test10本轮不运行。按原scene身份复核互斥，多个子图同原景不能跨池。不新切出更有利的val。

**L8：** 使用既有 `phase45_protocol_audit/scene_disjoint_manifest.csv` 和 `research_plans/protocol_data/l8_biome_usgs_shadow_status.csv`，沿Phase64的target_train/target_val及Shadows?=yes资格；test本轮不运行。按原scene分组，裁剪图不是独立场景。l8_biome与l8_biome_raw不算两个数据集。

- SPARCS当前入口是 `src/cloud_adapter/datasets/sparcs_manifest.py` 中的原RGB预览，L8是原manifest指向的RGB PNG。必须明确记录实际输入是什么；预览图不得写成原始反射率，也不能默认套CMC的TOA×255读取路径。
- 对SPARCS/L8，首轮冻结为既有RGB显示输入，保持原通道及归一化链；每张按历史val的512×512 Resize规则读取。训练和教师也使用同一512视野，水平翻转按固定表同步。不加PhotoMetricDistortion，不在本轮改原始多波段。此为新无标签试验的明确输入定义，不声称等同旧Phase64的随机原尺寸训练。
- 复用mmcv/mmseg原Resize的图像插值与数据预处理，标签仅在评价时按旧规则最近邻读取。先对原历史val按产品/成员名字排序的首图，用旧推理入口与本轮入口比较输入张量和Source输出；最大误差<=1e-6、pred一致。两池各取一个，不据效果挑样例。
- SPARCS Surface=water/snow/land/flooded，Cloud=cloud，Shadow=shadow/shadow-over-water。L8父类沿现有映射：clear→Surface，thin/cloud→Cloud，shadow→Shadow，fill→ignore。保留原生子类，评价中单列雪/水/薄云，不把Surface解释成裸地。
- 训练valid只来自图像与其已有文件级缺失规则，不打开mask填补有效区。无法从RGB可靠判定的nodata写清：不得把显示黑色无依据当无效。文件级元数据资格与标签像元有效性分开。

### 3．控制训练量，冻结图像日程

- SPARCS使用原target_train全部图像（预期60）。L8从原有资格的target_train中选最多100张作为小试验fit池：按scene排序，在每scene内按 `SHA256('phase317-fit:'+完整成员相对路径)` 排序，逐scene轮转各取一张，直至100张或候选耗尽。先锁清单，不看标签、Source预测或模型效果。
- L8的“取100张”只限制本轮小试验成本，不代表完整L8训练池，也不能把其结论与全量监督方法公平排名。列实际scene数和每scene张数，不把大场景拥有更多裁剪图当更多独立样本。
- SPARCS/L8分别按排序fit名单构造1000尝试步：NumPy PCG64(seed31700)，每步均匀抽一个fit索引，并独立抽水平翻转bool；保存完整日程及SHA。每池PL/CB用同一表，数据集之间不是同一组图像；不按伪标签非空情况替换步骤。
- 新两池val使用全部原target_val且符合已有完整标注资格的图像，先列实际数量/原scene支持。每池val最多1200图的本轮资源边界；若实际超过，不按模型表现抽样，交具体资源缺项。预期L8历史对应963图，以实际锁定清单为准。
- CMC继续用旧日程；SPARCS/L8日程与CMC不同，跨池只比较各自同协议下的增量，不能把绝对mIoU直接排成传感器优劣。

### 4．同一套伪标签，两种训练损失

- 新两池对冻结512输入各做一次Source前向，保存s/pred/valid及Q90掩膜。语义得分仍为原Mask2Former query聚合、排除no-object；q=s/sum(s)。valid、sum>1e-12且max(q)>=0.90接受argmax(s)，否则ignore255。不扫阈值、不新增教师、不开EMA。
- 先把全部fit伪标签与1000步每类像元数锁好，再训练。报告唯一接纳像元、每类比例、非空产品、实际非空步骤，以及每类出现步骤；不因Shadow很少就悄悄降低门槛。
- **PL：** 原 `CE(s,pseudo)` 对本步全部接纳像元取平均。
- **CB：** 对本步每个有接纳支持的类别，先单独平均其CE，再对这些非空类别等权平均：`L_CB=(1/|C_present|)*sum_c mean_{A_c} CE(s,pseudo)`。缺失类别不补0项，不重复采样，不按真实标签构造权重，不用全池真值类比例。仅改变loss平均方式，接纳集合和更新步骤与PL完全一致。
- 此CB只是明确的简单类均衡对照，不是声称复现某篇CBST论文。Shadow伪标签可能有噪声，增大其相对权重也可能伤害结果；不预先保证有效，更不称其为云特有新方法。
- 两新池均PL/CB各1000尝试，A为空跳过前向/优化，不补步。若某池更新很少、Shadow从未出现，照实交付有效支持，不改成“完成1000更新”。CMC-CB必须精确复现955更新/45跳过。
- AdamW仅含LoRA，lr1e-4/wd0.01/betas(0.9,0.999)/eps1e-8、恒定学习率、FP32/batch1。保持梯度穿冻结head；冻结参数/buffer初末逐张量一致，不加梯度裁剪或混合精度。
- CMC PL复用旧30201，不重训；CMC CB、SPARCS PL/CB、L8 PL/CB合计最多5个新增训练臂。没有单独“新方法”臂，不把CB本身包装成创新。

### 5．正确性检查、封存及评价

- 零LoRA与Source一致<=1e-6；参数数与梯度白名单通过，先门检后恢复初始/优化器/RNG。用手算小例验证CB的平均：类别像元数不同但各类CE相同应得到相同结果；只有一个类别时CB=PL；全ignore跳过。
- 每池PL/CB初始权重、数据、掩膜SHA与日程相同。门检不得读取目标参考。训练日志含每类数、各类平均CE、总loss、梯度范数和真实更新标志。
- 每臂固定初始及第1000尝试步末权重。CMC CB沿原4探针×0/100/500/1000，共16次；SPARCS/L8每臂仅初始与末步对按名字排序的首fit图保存无标签输出，共2次/臂，不加中间选模。不把探针当评价替代。
- 全部5个固定末步、代码/日程/伪标签及日志封存后，独立CPU进程才读取本轮参考。两新池Source评价缓存可提前生成，但不得在配置冻结前查看其参考指标；已有历史暴露照实披露。
- CMC沿原32组9窗完整评价。新两池沿上文冻结512视野，每图一次前向，不改成native滑窗或为更好分数更换拉伸。每个端点保存pred、valid及float32 Shadow评分softmax(s)[2]；无需持久保存全部val的三类s，门检样例与必要审计保留即可。
- 主表每池Source/PL/CB，三类池化IoU、按scene/group均mIoU、池化mIoU、Shadow宏/池化AP、宏召回/FPR、可算组数和支持。完整逐scene及native子类表另交。零union类NA规则沿当前研究，所有方法同支持；旧Phase64分数若协议不同只能历史附注，不能拿来代替本轮Source。
- 对Source→PL、Source→CB、PL→CB交27格与六类有向错误，边缘核对混淆。每池按本池PL定义D/B，分别交CB的D修复/残留/转Shadow、B保留/退回Shadow/转Surface以及D外新增错误。CMC D=506798、B=784478应复现；两新池不套用这个分母。
- 阴影召回或mIoU上升时，必须同时列Surface→Cloud/Shadow误报、Cloud纠正撤销和FPR代价；不能靠把更多像元判Shadow就宣布解决了问题。

### 6．这轮结果怎么决定下一步

- 这是一个seed的方向性筛查，不报告稳定跨seed优势。三个数据集独立列出，不能用大数据池的像元数淹没另一池失败，也不要求每个指标全赢。
- 若CB在多个池改善Shadow且代价可接受，首先说明简单类均衡已有作用；后续新方法必须超过这个对照，不能把CB改名投稿。
- 若CB只在CMC好、另两池反向或代价明显，记录实际失效位置；只有能指出新的具体原因和可检验修订时才设计一个候选，不继续无目标加模块。
- 若PL/CB因接纳规则导致新池有效训练几乎没有，结果属于支持不足，不能算验证了方法无效；但本轮不调阈值救回。先根据明确支持差异作一次项目决策，避免复制Phase302的问题。
- 初步“值得继续”由实际效应、代价、组间范围及任务需要共同判断，没有统一必须涨1pp/CI跨零即失败的门槛。若没有可用信号，收口本轮并保留报告，不自动加seed或另一种loss。
- 14天是本次新方法尝试的局部决策窗口，以第一次正式执行实际开始时间落锁；不是从过去phase倒算，也不是整个项目的日历截止。重复失败/等待时间照实记账，不用拆新phase自动重置窗口。到期汇总已有证据决定是否继续，不能靠窗口内没完成就宣布算法原理被否定。

## 三、交付什么

目录 `outputs/phase317/phase317_YYYYMMDD/`，摘要 `src/research_results/phase317_YYYYMMDD/`；日期用实际开始北京时间。RESULT.md两页内，开头用简单话回答：三份数据是否真正跑起来；均衡训练有没有帮助云影；为此付出什么代价；值不值得做一个新方法。

**表一：数据和真实训练支持（DATA_AND_SUPPORT.csv）。**

| 数据池 | 原scene/fit张数/val张数 | 输入来源/显示或TOA/尺寸 | 三类接纳数 | 类别出现步数 | PL/CB尝试/更新/跳过 | 旧使用与协议差异 |
|---|---|---|---|---|---|---|

**表二：三池主结果（METRICS.csv）。**

| 数据池/策略 | 组均/池化mIoU | 三类IoU | Shadow宏/池化AP | 宏召回/FPR | 可算组数/分母 | 相对Source、CB−PL差值 |
|---|---|---|---|---|---|---|

**表三：修复与撤销（DAMAGE_BENEFIT.csv）。**

| 池/scene | 本池PL的D/B | CB修复D/仍Cloud/转Shadow | CB保留B/退回Shadow/转Surface | D外新增Surface→Cloud/Shadow | 其他错误变化 |
|---|---|---|---|---|---|

**表四：原生类别与全部组（PER_SCENE_NATIVE.csv）。**

| 池/scene/原生类 | 参考支持 | 三策略父类预测分布 | Cloud/Shadow误报或漏报 | 组mIoU与union数 | 缺类/未完整标注说明 |
|---|---|---|---|---|---|

**表五：本轮决策（DECISION.csv）。**

| 问题 | 三池实际证据 | 代价/不确定性 | 普通均衡是否已有作用 | 若继续须解决的具体缺口 | 继续/收口及理由 |
|---|---|---|---|---|---|

另交5个新增初末LoRA、原Source引用锁、图像成员/划分/伪标签/日程SHA、初始与输入一致性门检、训练日志、评价缓存、完整27格/六向流、独立CPU复算、资源/失败及完整备份校验。旧目标监督端点不能进入本轮训练初始化。

预算：训练前锁定实际调用账。CMC-CB约955训练+288评价+16探针；SPARCS/L8各最多2000训练，教师fit最多60/100，Source与两个端点val各按实际图数计算。每池val<=1200；新增总前向含门检/构建/失败/恢复<=11000，GPU阶段合计<=12小时。顺序单GPU、启动查冲突；沿gzs cloud-lite-pt210、原FP32算子，零下载/升级。CPU<=16GiB，新持久产物含双机备份<=16GB；宿主空闲内存>=4GiB、磁盘>=8GiB，启动前按完整产物估算验证容量。原始图像/历史缓存只读引用，不整包复制。不能满足则交具体缺项，不改精度、删val或压缩评分精度凑预算。计算在gzs，本机仅编辑、SHA与备份。无需定时检查。本文件仅立项，不启动执行、不新增push/pull授权、不自动开始下一阶段。
