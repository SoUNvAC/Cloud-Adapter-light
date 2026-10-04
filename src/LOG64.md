# LOG64 — 异构标签空间的层级一致性参数高效域适配

## 2026-10-04 — Day 1 数据与协议冻结

- 目标：停止逐像素人工裁决路线，先核验是否能用公开数据构造 `surface_visible/cloud/shadow` 共享父类与数据集特定原生子类，不新增人工标签、不训练新模型。
- 改动：新增 Phase 64 预注册、四数据域机器可读 registry、无第三方依赖的协议审计及 4 个回归测试。父类 `Clear` 修订为语义明确的 `surface_visible`；snow/water/flooded 保留原生子类，Fill/Unlabelled 显式 ignore；L8 完整三父类监督固定只用 `Shadows?=yes`。
- 网络情况：本地首次下载因 Windows Schannel 凭据失败；获准后从 PANGAEA 官方地址成功取得 2.1 MB Deep Fmask 标签包。SHA256 为 `eb53c2dd795c1541afcef4f8cf5398953748883c82a79d9512dfa3fbb28e7bcd`；实测类别 ID 为 0–5，SAFE 清单为 validation 22、test 23。未下载 Sentinel-2 SAFE 原始影像。
- 止损线：Day 2 前必须有两个目标域 loader、数据 hash 和 scene 泄漏审计；Day 3 要求两个目标域同向、父类 mIoU 平均至少 `+2.0`、source 下降不超过 `1.0`、MsRE 新增参数不超过 `0.50M`、Shadow 不灾难性坍缩。未达即停止当前组合，不新增模块。
- 真实结果：registry 审计通过，4 个单元测试通过；公开候选为 CloudSEN12 High、L8 Biome、Deep Fmask、SPARCS。当前只有 L8 一个目标域 loader 已实现，`day2_two_target_execution_ready=false`；因此 Day 1 科学协议通过、两目标训练执行门未通过，本轮没有模型指标。
- 同步核验：提交 `be32d0c` 已 push 并由 gzs fast-forward pull；远端 `cloud-lite-pt210` 中 4 个测试与 registry 审计再次通过。只检查授权路径后确认 CloudSEN 与 L8 derived 数据目录存在，获准只读的 `/home/scv/shared/data/l8_biome_raw` 也存在；未访问其他共享数据目录。

## 2026-10-04 — Day 2 SPARCS 准备与路径边界事件

- 目标：把 USGS SPARCS 作为第二个可执行目标域，冻结 80 scene 的项目划分，并建立父类/native label loader；不训练模型。
- 改动：新增安全 ZIP/CRC、80 mask、shape、label、source-scene 与 60/10/10 scene-disjoint 审计；新增 manifest loader；新增按结构严格满足 `P(parent)=ΣP(native child)` 的共享父类/数据集条件子类 head 原型。远端与本地的协议、loader、概率恒等式和梯度测试均通过。
- 网络与真实观测：USGS 官方归档下载完成，大小 `1,555,733,976` bytes，SHA256 `5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a`。首次审计读到 400 个成员、80 个独立 Landsat scene、80 个 1000×1000 mask、标签仅 0–6；固定哈希划分为 60/10/10 scene，三个父类在每个 split 均有像元。target-test 的 native `flooded` 为 0 像元，后续不得为该 split 报告 flooded IoU。
- 权限事件：随后发现 `/home/scv/Cloud-Adapter-light/data -> ../shared/data/`，所以归档与解压目录实际解析为未授权的 `/home/scv/shared/data/sparcs`。发现后立即停止对该目录的读取、移动和删除；首次审计结果只作故障记录，不作为正式执行就绪证据。代码已增加 `allowed_root` 的 symlink-resolved containment 硬门，并把默认目录改到仓库实体目录 `phase64_data/sparcs`。
- 当前止损：在用户明确授权如何处置误落盘目录前，不再访问 `/home/scv/shared/data/sparcs`；正式 SPARCS 审计必须从官方 URL 重新下载到解析后仍位于 `/home/scv/Cloud-Adapter-light` 的路径并复跑。没有正式第二目标域前不得启动 Day 2 双域训练。

## 2026-10-04 — Day 2 基线实现与训练前硬门

- 目标：在不读取越界数据、不启动正式训练的前提下，把共享三父类 readout、full fine-tuning、标准 LoRA、MsRE 与 head-only 基线实现到可审计状态。
- 改动：Day 2 快速筛选固定为 RGB，避免把标签层级收益与六波段输入变化混杂；公共六波段移至过门后的 Phase 64B。新增 CloudSEN/L8 父类 dataset view、SPARCS loader、标准 DINO LoRA、显式 full-FT backbone、父类 checkpoint 转换器、2 target × 4 method 配置矩阵及权重/参数审计。转换器只移除 `decode_head.cls_embed.{weight,bias}` 两个四类不兼容张量，其余 379 个张量保留。
- 网络情况：USGS 官方 SPARCS 重下载在仓库实体目录续传到 `829,991,528 / 1,555,733,976` bytes 后，服务端连续返回 HTTP 500 或 TLS EOF；官方页面未发现第二下载镜像。保留部分文件等待同 URL 续传，未读取或利用共享区误落副本。
- 止损线：正式训练前必须取得 CloudSEN derived 与 L8 derived 的明确只读授权；SPARCS 必须在仓库实体目录完成 SHA256/CRC/scene 审计；配置、checkpoint 和可训练参数硬门必须全部通过。任一额外缺失/意外权重、MsRE 超过 `0.50M` 或数据路径越界均停止。
- 真实结果：配置矩阵 `8/8` 通过。Phase 22 输入 checkpoint SHA256 为 `d33a81337544e294a42007c6b353ee13c901ebb1e1f5261fd1d95a79e3888012`，转换输出 SHA256 为 `e518be667397ebb5a9f32781436bca6c2e0f22aa0c1ad7e6de9ad6448ac813b2`。模型实例化/加载审计通过；LoRA 可训练参数 `294,912`，MsRE `380,577`，二者训练名称与声明完全一致，MsRE 低于硬上限。尚未训练，因此没有 mIoU、Shadow IoU 或 source forgetting 结果。
- 权限阻塞：仓库 `data` 实际解析到 `/home/scv/shared/data`；CloudSEN 根为 `/home/scv/shared/data/cloudsen12_high_l1c`，L8 manifest 指向 `data/l8_biome`（实际为 `/home/scv/shared/data/l8_biome`）。此前只授权 `/home/scv/shared/data/l8_biome_raw`，故未读取这两个 derived 目录，训练保持暂停。

## 2026-10-04 — Day 2 数据解封、源基线与首个目标作业

- 目标：在新增的 `/home/scv/shared/` 访问授权下完成第二目标域硬审计、冻结 source-only 父类基线，并按一次一个作业启动目标域快速筛选；共享目录保持只读。
- 改动：从共享区已有完整 SPARCS 归档只读复制到仓库实体目录，修正 `*_qmask.tif` 与 `*_data.tif` 的精确配对规则；新增数据就绪审计、受保护训练 launcher、source retention 配置和显式父类 IoU metric。独立评估器改用 Pillow 保留 CloudSEN 调色板 PNG 的类别索引；OpenCV 会把该索引错误展开为三通道颜色。远端调色板回归测试 `2/2` 通过。
- 网络情况：未再次联网下载数据。只读来源 `/home/scv/shared/data/sparcs/l8cloudmasks.zip` 与仓库副本均为 `1,555,733,976` bytes，SHA256 均为 `5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a`；失败的部分下载文件独立保留，未覆盖或删除。
- 止损线：两个目标域相对 source-only 必须同向，父类 mIoU 平均提升至少 `+2.0`，source mIoU 下降不超过 `1.0`，MsRE 新增参数不超过 `0.50M`，Shadow 不得灾难性坍缩；target-test 继续封存。未满足即停止当前数据/骨干组合，不新增模块。
- 真实结果：SPARCS 正式审计通过：400 个归档成员、80 个样本/scene、60/10/10 split、mask 均为 1000×1000 且 ID 仅 0–6，manifest SHA256 为 `8b4305280331e42853e0cca726d54a1ca27feddc6f9e6d6c4fdfb10079353b99`。数据就绪审计通过：CloudSEN 8490/535/975、L8 `Shadows?=yes` 1560/963、SPARCS 60/10，路径、父类映射和三父类覆盖均合法。三类 source parent 训练 4000 step 完成，source-val mIoU `77.82`（surface/cloud/shadow `88.30/83.84/61.31`），checkpoint SHA256 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。
- source-only 冻结基线：CloudSEN source-val mIoU `77.85`、L8 target-val `58.34`、SPARCS target-val `57.71`；对应 Shadow IoU 为 `61.38/27.15/22.03`。评估张数为 `535/963/10`，L8 为 8 个 scene、SPARCS 为 10 个 scene，均完成 2000 次分组 bootstrap；没有读取 target-test。CloudSEN 缺少可恢复的 scene ID，因此其 535 单位 bootstrap 实际是 patch 级，不能表述为独立 scene 证据。
- 当前状态：`L8/shared_parent/seed64` 已由 source parent checkpoint 启动；iter 50 总 loss `15.2145`，各项有限，进程存活。CUDA/CuBLAS、grid sampler 和 cumsum 在 deterministic warn-only 模式下报告不可完全确定性，必须作为复现实验限制保留，不能宣称 bitwise deterministic。30 分钟心跳负责核对并串行推进矩阵。

## 2026-10-05 — Day 3 单 seed 快速筛选与止损判定

- 目标：完成 RGB、seed 64、4000 step 的 `L8/SPARCS × shared_parent/full/LoRA/MsRE` 冻结矩阵；每个真实最佳 target checkpoint 在同一 CloudSEN-val 535 张上复核 source forgetting，target-test 始终封存。
- 改动：新增受保护的 source-retention launcher 与最终汇总/止损脚本；所有训练、评估均有 PID、exit 0 与完成标记。独立评估器的 CloudSEN 调色板 PNG 读取错误已用 Pillow 索引读取修复，远端回归测试通过。没有修改数据、标签、冻结超参或已有 Phase 60–62 记录。
- 网络情况：远端 SSH 曾两次瞬时返回 `kex_exchange_identification: Connection closed by remote host`，重试后恢复；训练中 step 时间有波动，但项目内始终只有一个训练主进程，未发现 OOM、Traceback、NaN/Inf。共享目录仅只读使用，未修改或删除。
- 止损线：核心 MsRE 必须在两个目标域相对 source-only 同向提升、平均父类 mIoU 增益至少 `+2.0`、每域 source 下降不超过 `1.0`、参数不超过 `0.50M`；Shadow 灾难性坍缩预先操作化为任一目标 Shadow `<10` 或相对基线下降超过 `10` 点。所有日志 loss 必须有限，target-test 必须封存。
- L8 真实最佳结果（source-only `58.34`, Shadow `27.15`）：shared-parent `64.10/35.63`，source `63.21`；full `51.51/13.83`，source `38.21`；LoRA `62.80/35.71`，source `66.61`；MsRE `61.03/26.79`，source `77.81`。前三者分别 source 下降 `14.64/39.64/11.24`；MsRE 仅下降 `0.04`，目标 mIoU `+2.69`，Shadow `-0.36`。
- SPARCS 真实最佳结果（source-only `57.71`, Shadow `22.03`）：shared-parent `69.65/42.98`，source `69.92`；full `61.05/34.56`，source `50.54`；LoRA `68.82/43.82`，source `68.16`；MsRE `67.15/41.60`，source `77.81`。前三者分别 source 下降 `7.93/27.31/9.69`；MsRE 仅下降 `0.04`，目标 mIoU `+9.44`，Shadow `+19.57`。
- 判定：MsRE 两域同向，平均目标增益 `+6.07`，source 每域下降 `0.04`，可训练参数 `380,577`，Shadow 非灾难性坍缩；完整矩阵 loss 有限且 target-test 封存，所有预注册 Day 3 硬门通过，决策为 `continue_to_phase64b_common_six_band`。但这只是单 seed 点估计快速筛选，适配模型尚未做 scene bootstrap，CUDA warn-only 也非 bitwise deterministic；不得把它写成稳健性或统计显著性结论。
- 产物：最终机器可读汇总位于 `src/work_dirs/phase64_final_summary/summary.json`；MsRE 最佳 checkpoint SHA256 为 L8 `fcfa934e37e1ed1569e4eded2547ee999626840a6e03e9a58395ab6d5e8a7d57`、SPARCS `88c925370b5fa08345c1b98578f657b9559eaec8e428348bd6895519ffe49fce`。
