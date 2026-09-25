# Phase 62 实验记录

## 2026-09-25：部分标签建模审计启动

- 目标：验证 `Shadows?=no` 图像中 raw ID0 采用集合标签 `{clear, shadow}` 是否优于直接 ignore，同时保留 thin/thick 普通监督。
- 改动：不增加网络参数或结构模块；三组共用 Phase52 冻结 LightCloudHead 与既有 380,577 参数 target adapter、同一初始化、优化器、增强、4000 steps 和 seed 62/63/64。仅改变 53 张 `Shadows?=no` 图像 raw ID0 的数据/损失口径。
- 对照：`complete12`；`ignore65`（12 complete + 53 thin/thick，ID0 ignore）；`partial65`（12 complete + 53 thin/thick，ID0 使用 `-log(p_clear+p_shadow)`）。
- 评估：target-val 仅使用 `Shadows?=yes` 的 963 张、8 scenes；source-val 535 张按既定 disabled-target 路径检查遗忘；target-test 与 CloudSEN internal-test 继续封存。
- 止损线：相对 `ignore65`，partial 组 Thin/Shadow 调和均值必须提高，Thin 与 Shadow 的三 seed 均值都不得下降；分层 paired scene bootstrap 95% CI 下界必须大于 0 且至少 6/8 scenes 同向；三个 partial seed 的 source mIoU 均须 ≥73.93；non-clear mIoU 必须提高。任一失败即停止复杂标签转移矩阵路线。
- 结果：尚未产生，严禁提前填写。
