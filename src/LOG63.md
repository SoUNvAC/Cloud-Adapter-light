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
