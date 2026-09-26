# Phase 61D2 实验记录

## 2026-09-26：校准协议与执行前审计

- 目标：重做校准后的盲法复核；reviewer 对原标签、模型和方法盲态，但获得 B1–B11、推荐合成、太阳几何和必要元数据；40 个校准单元不进入统计，160 个独立单元与 50 个未见确认单元进入主复核。
- 改动：新增 61D2 复核包生成、校准锁、分歧专用第三方裁决和评分脚本；支持 `uncertain`、`unobservable_nodata` 与最多两个语义类的集合标签；硬门按 κ/AC1、Thin/Shadow reviewer IoU 和原标签—裁决标签 IoU 自动执行。
- 网络/读出：不训练、不修改任何模型；只做确定性抽样、11 波段证据页渲染和人工复核统计。确认集不按模型错误或置信度筛选。
- 数据口径：旧 200 个错配分层单元中确定性抽取 40 个校准单元，剩余 160 个作为独立复核；另从旧包未出现的 `Shadows?=yes` target-val 图块按 Thin/Shadow、scene/biome、boundary/interior 平衡抽取 50 个确认单元。校准集被评分器结构性排除。
- 止损线：校准锁定前不得开始主复核；A/B 原始判断必须保留，第三位专家只裁决真实分歧；κ 或 AC1 任一低于 0.40 时四类 mIoU 不得作为主指标；Thin 或 Shadow reviewer IoU 任一低于 0.30 时终止四类硬标签主线；不得以旧 RGB 图或 Phase54 三个辅助光谱通道冒充 B1–B11。
- 真实状态：本地 4 项协议单元测试、语法检查和 11 波段证据页合成渲染 smoke test 通过。远端仓库内 `data` 为指向 `/home/scv/shared/data` 的符号链接；完整 B1–B11/MTL 不在已授权的 `/home/scv/Cloud-Adapter-light` 实体目录内，现有 Phase54 cache 也不含全部波段。因此尚未读取外部 raw-root、尚未生成正式复核包、尚无人工结果或 κ/AC1/IoU；等待明确只读授权或将原始数据放入授权目录。

## 2026-09-26：只读授权后生成正式复核包

- 目标：在不泄露原标签、模型输出、方法或抽样来源的前提下，为两位 reviewer 提供完成校准和独立复核所需的完整多光谱证据。
- 改动：用户明确授权只读访问 `/home/scv/shared/data/l8_biome_raw`；生成过程只写 `/home/scv/Cloud-Adapter-light/src/work_dirs/phase61d2_calibrated_review`。原始目录含 96 个 scene 主栈，每个主栈 11 波段，未尝试写入。
- 网络/同步：本地提交 `96759c5` 已 push；gzs 最初连续出现 GitHub GnuTLS/SSH 连接失败，本轮重试后成功 fast-forward 至同一提交。远端使用 `cloud-lite-pt210`，4 项协议测试通过；不训练或修改模型。
- 数据与产物：生成 40 个校准单元、160 个独立单元和 50 个确认单元，共 250 张证据页；确认集使用 50 张互不重复且未出现在旧 61D 包中的图块，原始 Thin/Shadow 各 25。主复核 210 个单元的 biome 数为 snow_ice 44、shrubland 22、grass_crops 40、barren 36、wetlands 38、forest 30。A/B 的校准和主复核 CSV 均保持全空白。
- 质量核验：250/250 PNG 均为 1040×1210 RGB，ID 与 sealed manifest 一一对应；三张跨校准/主复核抽样证据页目视通过；ZIP 有 257 个 reviewer 可见成员和 250 张 PNG，不含 sealed manifest。ZIP SHA256 `df772757afedebab0a82599d59a10ad51bbd2e7b3858721f4862cf0fcb04af49`；sealed manifest SHA256 `7c9468778ccb44127610918ae537a64d7e1e0af95469c403a0c6c1453d587c77`；packet summary SHA256 `0c2f36c49ceeea87b43549d83c0677138adfdd6714b3edc37fe965a4873ea402`；manual SHA256 `cfa8ec5b65474dc4e6d4402cc9db0838db237afd715f081ff88f41cdb068d172`。
- 止损线与结果：状态为 `awaiting_calibration_lock`，`human_results_available=false`。40 个校准单元完成共同讨论并锁定前不得开始主复核；当前没有人工标签、κ、AC1、reviewer IoU、原标签—裁决标签 IoU或四类主线结论，严禁提前填报。
