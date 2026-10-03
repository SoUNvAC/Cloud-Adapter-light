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
