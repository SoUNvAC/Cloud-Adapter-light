# LOG64 — 异构标签空间的层级一致性参数高效域适配

## 2026-10-04 — Day 1 数据与协议冻结

- 目标：停止逐像素人工裁决路线，先核验是否能用公开数据构造 `surface_visible/cloud/shadow` 共享父类与数据集特定原生子类，不新增人工标签、不训练新模型。
- 改动：新增 Phase 64 预注册、四数据域机器可读 registry、无第三方依赖的协议审计及 4 个回归测试。父类 `Clear` 修订为语义明确的 `surface_visible`；snow/water/flooded 保留原生子类，Fill/Unlabelled 显式 ignore；L8 完整三父类监督固定只用 `Shadows?=yes`。
- 网络情况：本地首次下载因 Windows Schannel 凭据失败；获准后从 PANGAEA 官方地址成功取得 2.1 MB Deep Fmask 标签包。SHA256 为 `eb53c2dd795c1541afcef4f8cf5398953748883c82a79d9512dfa3fbb28e7bcd`；实测类别 ID 为 0–5，SAFE 清单为 validation 22、test 23。未下载 Sentinel-2 SAFE 原始影像。
- 止损线：Day 2 前必须有两个目标域 loader、数据 hash 和 scene 泄漏审计；Day 3 要求两个目标域同向、父类 mIoU 平均至少 `+2.0`、source 下降不超过 `1.0`、MsRE 新增参数不超过 `0.50M`、Shadow 不灾难性坍缩。未达即停止当前组合，不新增模块。
- 真实结果：registry 审计通过，4 个单元测试通过；公开候选为 CloudSEN12 High、L8 Biome、Deep Fmask、SPARCS。当前只有 L8 一个目标域 loader 已实现，`day2_two_target_execution_ready=false`；因此 Day 1 科学协议通过、两目标训练执行门未通过，本轮没有模型指标。
