# Phase 62 实验记录

## 2026-09-25：部分标签建模审计启动

- 目标：验证 `Shadows?=no` 图像中 raw ID0 采用集合标签 `{clear, shadow}` 是否优于直接 ignore，同时保留 thin/thick 普通监督。
- 改动：不增加网络参数或结构模块；三组共用 Phase52 冻结 Mask2Former head 与既有 380,577 参数 target adapter、同一初始化、优化器、增强、4000 steps 和 seed 62/63/64。仅改变 53 张 `Shadows?=no` 图像 raw ID0 的数据/损失口径。
- 对照：`complete12`；`ignore65`（12 complete + 53 thin/thick，ID0 ignore）；`partial65`（12 complete + 53 thin/thick，ID0 使用 `-log(p_clear+p_shadow)`）。
- 评估：target-val 仅使用 `Shadows?=yes` 的 963 张、8 scenes；source-val 535 张按既定 disabled-target 路径检查遗忘；target-test 与 CloudSEN internal-test 继续封存。
- 止损线：相对 `ignore65`，partial 组 Thin/Shadow 调和均值必须提高，Thin 与 Shadow 的三 seed 均值都不得下降；分层 paired scene bootstrap 95% CI 下界必须大于 0 且至少 6/8 scenes 同向；三个 partial seed 的 source mIoU 均须 ≥73.93；non-clear mIoU 必须提高。任一失败即停止复杂标签转移矩阵路线。
- 结果：尚未产生，严禁提前填写。

## 2026-09-25：预检通过并启动 3×3 seed

- 目标：在训练前验证冻结选择集、USGS 状态、三组像元映射、部分标签公式、参数公平性与梯度可达性。
- 改动：新增无参数 `PartialLabelFrozenHeadEncoderDecoder`；常规监督保持原生 Mask2Former loss，只有保留值 254 的像元追加 `-log(p_clear+p_shadow)`。预检首次发现有效 head 为 Mask2Former，改用仓库 Phase53 同口径的 dense semantic prediction；另修复两处仅影响审计器的 MMEngine lazy/serialized dataset 读取问题。所有修复均在训练启动前完成。
- 网络/运行：本地代码经提交 `b57fe10`、`90fc20c`、`89120aa`、`0afaca7` push；gzs 多次遇到 GitHub GnuTLS 非正常终止，最终以单次 HTTP/1.1 pull 成功同步。远端环境 `cloud-lite-pt210`，串行训练主 PID `449138`，避免九轮并发争用 GPU。
- 数据：manifest SHA256 `885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b`；选择集 SHA256 `564ad4e2c27d2948c725cf4c5c482159888c2324fe0ab4bdeee25fe774cebc55`；12 张 complete 与 53 张 `Shadows?=no` 精确复现。53 张 no 图像含 raw ID0/2/3 像元 7,051,817/2,295,161/4,546,654。
- 止损线：训练前必须满足组规模 12/65/65、raw ID0 映射 0/255/254、无新增可训练参数、解析损失为 `log 2`、真实模型反向梯度有限且可达原 adapter；任一失败不得训练。
- 结果：全部预检门通过。三组模型配置完全相同，可训练参数均为既有 380,577；均匀四类概率下 `{clear,shadow}` 部分损失为 0.69314718；真实 smoke partial loss 0.43044755，21 个既有可训练张量获得有限非零梯度。已启动 `complete12/seed62`，100/4000 step 时总损失 20.8223、partial loss 0（符合该组无部分标签像元），训练与最终指标尚未完成。

## 2026-09-26：3×3 seed 完成，部分标签继续门失败

- 目标：比较 `complete12`、`ignore65`、`partial65` 的三 seed target-val、source-val 与 scene bootstrap，决定是否发展复杂标签转移矩阵。
- 改动：无新增模型或训练改动；九轮均完成 4000 steps，随后逐 checkpoint 评估 963 张 `Shadows?=yes` target-val 与 535 张 source-val，并执行 10,000 次 seed×scene 分层配对 bootstrap。target-test 与 CloudSEN internal-test 未读取。
- 网络/运行：远端串行运行于 `cloud-lite-pt210`，完成时间 `2026-09-25T23:57:02+08:00`；九个 `TRAIN_COMPLETE`、总完成标记和九份 evaluation 均存在。训练/评估日志未发现 traceback、OOM、NaN 或 Inf；partial 组三轮 `partial.loss_set_nll` 均为有限非零值。
- 结果（mIoU/Thin/Shadow/弱类调和均值/non-clear mIoU，三 seed 均值）：`complete12`=38.905/25.306/11.261/15.563/26.423；`ignore65`=42.416/33.320/3.189/5.786/32.764；`partial65`=42.437/30.774/1.966/3.673/30.916。既定 disabled-target source mIoU 三组各 seed 均为 73.980，遗忘门通过。
- 相对 `ignore65`：partial 的 mIoU +0.021，但 Thin -2.546、Shadow -1.222、调和均值 -2.113、non-clear mIoU -1.848；Clear 单独 +5.629。三个配对 seed 的调和均值差为 -6.057/-0.580/+0.298，明显不是弱类一致改善，而是近乎完全由 clear 类抵消。
- scene bootstrap：弱类调和均值配对差均值 -2.091，95% CI [-6.132, 0.177]，改善概率 0.0695；8/8 scenes 的三 seed 平均差均为负，稳定方向与继续门相反。
- 止损线：调和均值提高、Thin 不降、Shadow 不降、bootstrap CI 正向、至少 6/8 scenes 正向、non-clear 提升和“非 clear-only”七项均失败；只有 source 遗忘、运行完整性与封存协议通过。
- 判定：`stop_complex_label_transfer_matrix`。当前 `{clear,shadow}` 部分标签基线无效，不发展复杂标签转移矩阵；不得用 +0.021 mIoU 宣称改善。
- 产物：`preflight.json` SHA256 `d56086bfe686e836a3f7d7c06b8b96f6dd3d7842726ddd827d60684033098036`；`summary.json` SHA256 `2244e5edc6d3db55f763997a60545b6325f0730a8fc4e8959562ca5d1284af45`；`evaluation_console.log` SHA256 `e9725b0207ca86731d0f9fb0f0cf526fce20c66e7d435090a78ac16171d37e1b`；`phase62_console.log` SHA256 `205cba49178074380a02010a6e3cc7a3df16ffc22f63812bf6a910da3a27b07a`。最终结构、样本数、运行数和决策断言复核通过。
