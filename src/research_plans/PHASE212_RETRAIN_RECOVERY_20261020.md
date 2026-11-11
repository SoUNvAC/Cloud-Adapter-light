# Phase212-R：两个目标解码头的节省存储重训

用户在2026-10-20明确选择重训，覆盖Phase212原件恢复要求和“零训练”限制，仅针对缺失的L8／SPARCS shared-parent两个头。原Phase212门检受阻交付只读保留，新结果不是原SHA恢复。

- 复用已核验三父类Source SHA `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`，原DINO初始化和cloud-lite-pt210 runtime。
- 两个目标沿用Phase64 target shared_parent原配置、seed64、AdamW／原调度、batch1，各4000步；每500步完整target-val按原mIoU选模型。L8 train1560／18scene，val963／8scene；SPARCS train60／60scene，val10／10scene。只读这两个train／val，禁止test。
- 完整backbone、Cloud-Adapter及其他非head组件冻结eval；逐张量核验Source初始化及完整非head参数／buffer训练前后不变。只允许decode_head参数训练，预期1,316,516参数。
- 更改仅持久化：最佳checkpoint只存完整decode_head state（含buffer）和Source／完整冻结状态哈希、迭代、选模信息；不保存optimizer／scheduler／重复backbone，不保存周期权重。只替换本次运行自身上一个best，不删除历史文件。可按Source+head恢复完整模型。
- 训练阶段预计8000训练前向＋8×(963+10)=7784验证前向，共15784；训练阶段上限16000调用，4h累计执行时间（失败消耗计入），GPU20GiB／CPU16GiB，本次新增持久≤500MB、宿主物理盘余量≥4GiB。预检所有Windows固定卷最低余量约58.6GB，具体值封存在预检；运行中只以新目录字节增量保护，不将WSL guest余量冒充宿主余量。
- 串行执行两个目标，不依结果决定第二域。失败确认退出并保留账本，不自动重训。不改精度、支持、seed或训练步数凑预算，不下载／升级环境。
- 新目录`/home/scv/shared/phase212_retrain_20261020`，只读旧work_dirs原件与清单，不同步work_dirs。代码本地检查，SHA封存复制；不自动Git push。
- 训练完成先封存新head SHA和精确端点验证结果，再接续原Phase212固定α=0.25／0.5／0.75的源／目标评价。原16000调用／4h插值评价预算独立保留，不将新端点要求为旧64.10／69.65等历史值，也不将重训差异归因插值。旧结果仅历史参照。
- 不建立定时任务，不持续SSH轮询，不启动其他phase；后台串行worker保存状态及退出码。原协议所有标签映射、数据边界、bootstrap与报告口径继续有效。
