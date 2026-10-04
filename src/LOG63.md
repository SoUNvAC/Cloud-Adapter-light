# Phase 63 实验记录

## 2026-09-29：选择性多粒度可行性协议、工具链与scene冻结预检

- 目标：不训练新分割网络，先检验“现有模型能否识别哪些中心像元具备四类细分条件、哪些只能退回粗粒度或拒识”。冻结Phase61D2的210个主复核点、79条缺失理由、snow/ice分层和Phase55模型对；160项仅用于风险开发，50项仅允许最终确认。
- 改动：新增Phase63预注册、scene锁与泄漏审计、现有checkpoint不可靠性证据导出、逻辑回归/固定GAM/isotonic风险评分、scene bootstrap与风险—覆盖曲线；新增无参数层级概率/router原型，不注册训练配置、不含深层risk MLP。四类模型输出顺序固定为`clear/thick/thin/shadow`；单四类softmax的层级恒等式明确标为不可用，只有独立coarse概率才能产生层级冲突特征。210项均按黄色角标中心像元评价，禁止称为dense mIoU。
- 数据冻结：sealed manifest SHA256 `7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77`；source-only Phase22 seed42 iter40000 checkpoint SHA256 `d33a81337544e294a42007c6b353ee13c901ebb1e1f5261fd1d95a79e3888012`；adapted Phase52A-fix seed52 iter1000 SHA256 `9655ee83c63840ecb31351efef0582676529c318fd0d1eca5fd0929db702eb2e`。最终210项中明确四类162项（dev126、confirmation36）；其余48项不强制映射到四类，而作为细粒度不可判/失败保留。79条裁决理由保持0/79，未补写。
- 网络/运行：只通过ssh gzs在授权仓库内只读核验已有checkpoint并同步真实sealed manifest；同步代码时两次SSH连接在密钥交换阶段被远端关闭，第三次恢复后正常fast-forward。未读取新数据、未下载第二域、未启动GPU推理或训练。本地与远端`cloud-lite-pt210`各20项协议/原型测试及语法检查通过；远端复现同一scene审计哈希且未生成split lock。用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`及`outputs/`改动未覆盖。
- scene真实结果：现有development/confirmation都来自同一8个scene，重叠8/8；各scene的dev/confirm数为14/8、14/8、14/8、21/8、32/6、26/4、32/4、7/4。snow/ice 44项原样保留（28/16）。审计按规则fail closed，`phase63_split_lock.json`未生成；审计产物SHA256 `cbeb56b9b2964110bb5162e23e78127f9ff1469162f1c31e536784ebc9b11ee2`。
- 止损线：在合法scene lock出现前，禁止导出带confirmation标签的证据、拟合风险模型或计算63C AUROC/coverage/bootstrap。因此本轮没有AUROC、CI、覆盖率或生死门结果，严禁填报。保持旧50项角色不变与scene零泄漏在当前210项上不可同时满足；优先方案是从此前未见scene取得新的确认集，备选是明确废止旧160/50角色后按完整scene重划，但旧50项不能继续称为未触碰确认集。
- C组状态：层级合并与外部风险router原型为无参数模块，fine/coarse/abstain、非有限风险拒识和独立coarse冲突测试通过；没有训练或性能结果，不能称为方法有效。
- D组状态：资料审计推荐Deep Fmask Sentinel-2雪冰/高山域作为第二标注域；它有45个scene和原生clear-land/cloud/shadow/snow/water标签，可做三类/二类与snow压力测试，但没有thin/thick像元真值、与CloudSEN同为Sentinel-2，不能称第二传感器或四类验证。Deep F-Mask Stage-4仅是论文锚点，未下载数据、未复现；报告SHA256 `8257076fb35f5f313b02c04374a0f19d36ab9b71d28890445ee73d4e631d06d9`。
- 当前判定：`blocked_before_phase63c_by_scene_leakage`。这不是选择性多粒度命题已经失败，也不是已经通过；只是现有确认设计不能给出无泄漏判定。不得以新模型或注意力模块绕过该数据门。

## 2026-09-29：新增scene独立人工确认包（等待复核，未运行63C）

- 目标：在任何风险模型拟合前修复旧160/50的scene泄漏。旧50项不再承担Phase63最终确认角色；新建两个预注册队列：32项严格`target_val`确认队列，以及18项Phase50零选中patch的snow/ice训练候选scene压力队列。后者必须单列，不能伪称严格未见验证集。
- 改动：新增`prepare_phase63_human_confirmation.py`及3项抽样/防泄漏测试，并让既有61D2图板渲染器接收阶段标题而不改变默认行为。抽样是像素标签盲和模型盲，但明确使用`Shadows?`及biome协议元数据分层；patch和中心坐标均由冻结输入与固定salt确定。预注册与代码提交为`af94a04`。
- 数据与结果：确认集共50项、11个scene；32项来自8个此前未进入旧160项、也未进入Phase50选中训练scene的`target_val Shadows?=no` scene，18项来自3个Phase50零选中patch的`target_train snow_ice` scene。与旧development scene重叠0，与Phase50选中训练scene重叠0，`target_test`读取0。reviewer包含40张旧校准图和50张新确认图，共90张1040×1210 RGB图；所有新图展示B1–B11、推荐合成、太阳方位和必要元数据。
- 盲态核验：ZIP共98个成员；5份待填CSV分别为40/40/50/40/50行，除`tile_id`外全部为空；包内无sealed manifest、原标签、模型输出或风险分数。reviewer ZIP SHA256 `dc16956f960c4d7e86329d41b06ddd98b35ce26311cc1359b3965fd8cdafab85`；sealed manifest SHA256 `59f64b9063c37c2de367481dc8b3cc3be08c3ad8df50d5a031338650545fb3e1`；packet summary SHA256 `24ef62f883b3e66807bf3c5fe7be778f641ecfd06499c7b60c96e2c0fafcb828`。抽查snow/ice及非snow图板，波段、合成、中心标记和方位图可读。
- 网络/运行：本地30项相关测试通过；代码push后，gzs授权仓库fast-forward到`af94a04`，远端`cloud-lite-pt210`中22项工具协议测试和8项无参数router测试通过。只读访问已授权的`/home/scv/shared/data/l8_biome_raw`生成图板，写入仅发生在远端仓库及本地仓库的Phase63 work dir；通过SSH/SCP同步成功。未启动GPU推理或训练，未改写用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`或`outputs/`。
- 止损线：当前状态为`awaiting_new_reviewer_calibration`，不是63C通过。A/B必须先独立完成40项校准并讨论记录共识，再独立完成50项确认；第三专家只裁决真实分歧。两份原始判断、裁决和哈希冻结前，禁止查看确认标签拟合风险、计算AUROC/coverage/bootstrap或启动新网络。最终必须分开报告32项严格确认队列与18项snow/ice压力队列；不得用合并50项冒充单一总体的无偏发生率。

## 2026-10-03：校准读出与评价单位缺陷修正

- 目标：核验新Phase63校准是否完成，并在继续confirmation前检查低一致率是否混入界面定位歧义。标注程序未覆盖根目录模板，真实完成记录位于`.annotation_history`最新快照；A、B与consensus各40行，其中同一第40项三方均为空，按用户既定规则记为纯色/无有效信息，39项可评价。
- 真实校准读出：A/B exact及compatible agreement均为9/39（23.08%）；Cohen κ `0.1308`，Gwet AC1 `0.1253`；Thin membership IoU `16.67%`，cloud-shadow `41.67%`，thick `0%`，clear `14.29%`。30项分歧中13项涉及`boundary_mixed`、4项为cloud-shadow↔terrain/water-shadow、2项thin↔thick、1项thin↔cloud-shadow。A使用`boundary_mixed` 1次，B使用12次。讨论后consensus与A一致28/39、与B一致17/39，不能当作独立一致性证据。A/B/consensus快照SHA256依次为`09ced51a68c40a342264132b7dce2ab58f40ac892b73ebdfe143c856a7ca47e2`、`1d1b43e21893cd9ae4889c5ffb2a52cc69272bfa9f2f6880407ae572ac789ff3`、`e072d0fd5c09f31ada811c3285b10ab7dd5fd4ef00b731db119be674e057194c`。
- 关键修正：旧图中文字虽称待判像元，但黄色角标在源分辨率上实际围住约20×20像元，不能唯一指定评价单位。因此上述低κ同时受目标定位歧义和语义歧义影响，不能全部归因于细粒度标签不可复现；Phase61D2旧数字不删除，但后续引用必须附此限制。
- 改动：新增16项精确像元UI pilot生成与评分工具、4项测试及独立手册。页面分离192×192上下文与21×21最近邻像元视图；空心框只包围一个像元且不遮挡内部。先填目标可定位性，再填可辨识性；只有可细分时才填语义类别，并记录耗时。pilot按8个biome各2项抽取此前未人工复核、未被Phase50选中的target-train patch，排除所有既有复核scene和target-test；只用于UI，不进入模型或确认统计。
- 网络/运行：本轮截至记录时仅完成本地语法检查及26项Phase工具测试，未访问远端原始数据、未生成pilot图、未启动GPU或训练。用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`与`outputs/`未改动。
- 止损线：正式confirmation继续暂停。UI pilot须由两人独立完成且每图只看一次；双方可定位率均≥95%、可辨识性exact≥70%且AC1≥0.40、双方均判可细分至少6项且其语义exact≥50%才允许重做正式confirmation。新旧项目难度不同，前后差异不得宣称为UI改动的因果效果；pilot尚无人工结果或Phase63生死门结果。

## 2026-10-03：精确像元 UI pilot 包生成完成（等待独立复核）

- 目标：以最小工作量检验“唯一中心像元 + 上下文/局部双视图 + 可辨识性先行”能否消除目标定位歧义；不训练、不评价模型、不读取confirmation标签。
- 改动与产物：代码提交`c643977`在gzs fast-forward后，以`cloud-lite-pt210`和只读原始波段生成16项pilot。8个biome各2项、8个此前未人工复核的target-train scene；与既有复核scene重叠0、与Phase50实际选中patch重叠0、target-test读取0。每页含192×192上下文、4组21×21合成和11个21×21单波段最近邻视图，显示精确`(x,y)`；空心框只包围一个源像元且中心内部可见。
- 核验结果：reviewer ZIP共23个成员、16张1040×1210 RGB图、A/B各16行空白表；包内无sealed manifest、原标签、模型预测或risk score。目视抽查普通biome与snow/ice页面，像元格、中心框、十字线、波段和太阳方向可读。ZIP SHA256 `3ea92af06157c405529e202768ce1b4e83919312dcd5f22f5b73ceb9912ff568`；sealed manifest SHA256 `7503977e55cd35be3c30c3c376efd2f51b09907ebd95df450bfa88da07153627`；packet summary SHA256 `bbbe6e3d47fd124f864f9831f13a6553af80a4afff165bf7338417de80ef1298`。
- 网络/运行：首次SSH连接在密钥交换阶段被远端关闭，重试后正常；远端4项pilot测试通过并完成生成，SCP同步成功。只读访问`/home/scv/shared/data/l8_biome_raw`，写入仅发生在授权仓库；无GPU推理或训练。用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`和`outputs/`未覆盖。
- 止损线与当前结果：状态为`awaiting_independent_review`。当前只有UI与盲态核验结果，没有人工一致率、耗时、模型AUROC或Phase63生死门结果。A/B必须独立、每图仅一轮并在原始CSV冻结前不讨论；pilot通过预注册UI门后才允许重做正式confirmation。

## 2026-10-04：semantic-coverage exact-pixel UI pilot 重新预注册并生成

- 目标：废止无语义哈希生成的16项UI pilot，暂停原随机中心50项confirmation；以原标签仅作sealed抽样层，生成20项只检验UI与标注流程的精确像元pilot。固定配额为clear/thin/thick/cloud-shadow interior各4项及thin↔thick、cloud↔shadow、clear↔shadow边界2/1/1项；优先不同scene、每scene最多2点、每patch最多1点、每个interior类至少3个scene。
- 改动：新增20项生成、冻结、评分与测试工具及操作手册；中心像元必须距任意标签边界`>3 px`，在不降低scene覆盖等硬约束时确定性优先`>=5 px`；边界点来自预注册类别对的真实界面。可观测性门固定为目标B2/B3/B4有限非零且居中192×192上下文RGB有效比例至少50%。纠正原始L8 fixedmask映射为`0=Fill,64=shadow,128=clear,192=thin,255=thick`，不得与转换后的四类ID0混淆。
- QA过程：首个候选包因把原始Fill=0误当clear而出现3/20张纯黑页，已标记`qa_rejected_before_review`并隔离；第二包虽满足`>3 px`硬门，但在`>=5 px`偏好尚未成为确定性审计字段时即被`qa_superseded_before_review`隔离。两包均未交给reviewer、不得评分。旧16项标为`superseded_before_review`；旧50项保持未读取并标为`suspended_not_formal_confirmation`。
- 真实结果：最终20项覆盖16个不同scene，最多2点/scene、1点/patch；四个interior名义类各覆盖4个scene，边界配额严格为2/1/1；16个interior中15个距离`>=5 px`，其余1个距离`3.606 px`，仍满足`>3 px`硬门；边界点最大类别对距离0 px。最小RGB有效上下文比例`0.6106`；20页均为1040×1210 RGB，无低信息/纯色页，最少上下文颜色数66。target-test读取0、模型数组读取0、与既有复核scene/patch重叠0。
- 产物核验：A/B CSV各20行且除ID外全空，ID与20页完全一致；ZIP共27个成员，内部SHA清单全匹配。最终reviewer ZIP SHA256 `bdae9a616e40f894b7e780672d70467e72e94b456094bd7bb8b5b2d76446b74a`；sealed manifest SHA256 `38e1168cf3a34636debdccb89d2b1b1e54b0a52d1d58cc0020937e82f0678fd6`；packet summary SHA256 `21cde11833697ec187a600b4712462fa49158c8ed04257a30d9bfa4ee77fea7`。
- 网络/运行：本地修改均先测试、commit和push，gzs授权仓库逐次fast-forward；远端固定使用`cloud-lite-pt210`，最终8项工具测试通过。只读访问授权的`/home/scv/shared/data/l8_biome_raw`，写入仅发生在本地/远端仓库Phase63 work dir；未训练、未启动GPU推理，未覆盖用户已有`LOG60.md`、`LOG61.md`、`LOG62.md`或`outputs/`改动。
- 止损线：本20项只能检验UI与流程。A/B必须独立完成；两人可定位率各至少19/20、零坐标/页面/格式错误、可辨识性exact至少70%、interior中双方均判可细分至少10项且其语义exact至少60%才通过。只报告名义层原始计数；A/B原始判断冻结前禁止解封sampling字段。当前状态`awaiting_independent_review`，没有人工一致率、类别准确率、模型AUROC、coverage或Phase63C生死门结果。

## 2026-10-04：semantic-coverage UI pilot 双评冻结与评分

- 目标与冻结：A/B各完成20项。格式校验发现A最初有3项`target_locatable=no`但仍填`mixed_boundary`，由reviewer自行修正为`no`且后续判断留空后再次校验通过；程序未替改原始判断。nominal层解封前冻结A/B SHA256为`0ed3a23d7f93d3f1d4e102cb221153aaa8c0ddd418a0ad7a7e199b46e92d73bb`/`e4a165894e36e46e7aac0abd6f96eb09c86f3373589893143436b76b3235a63f`，lock SHA256为`6b43b587e33fbc46b347713a4178984b11f8e681c4b3849d74bce88a27ca862f`。
- 真实结果：A/B可定位分别为17/20与20/20，共同可定位17项。可辨识性exact为15/17（88.24%），Cohen κ `-0.0625`、Gwet AC1 `0.8677`；负κ来自几乎全部选择`fine_classifiable`的极端边际分布，不能脱离exact/AC1解释。interior中双方均判可细分15项，语义exact为11/15（73.33%），κ `0.6429`、AC1 `0.6467`。平均耗时A/B为14.28/28.92秒。
- 名义层原始计数：A在clear/shadow/thick/thin四个interior层的可细分数为4/3/4/4，另1个shadow为证据不足；B四层均为4。A的语义计数依次为`clear2+haze1+shadow1`、`shadow3`、`thick4`、`haze1+clear1+thick2`；B依次为`clear4`、`shadow3+terrain/water-shadow1`、`thick4`、`haze3+clear1`。A有3/4个名义boundary项不可定位，B的4个boundary项为2个可细分、2个证据不足。reviewer–nominal-label concordance为A 9/15、B 11/16，仅是描述性一致，不是人工准确率。
- 止损与判定：六个预注册门中五个通过；唯一失败项为A可定位17/20低于19/20，因此`ui_pilot_pass=false`。本轮只说明当前UI/字段解释流程尚未稳定通过，不证明四类标签不可复现，也不评价模型。不得据此启动正式confirmation或Phase63C风险模型；metrics SHA256为`f163c36e903a2a6286682da4b31b936a32a08b2c0d3268aacdc22099b479e61e`。
