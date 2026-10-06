# Phase 64 RGB 补评与多 seed 复核

## 目标

不增加人工任务、不进入 64C。先用现有 L8 / SPARCS checkpoint 统一复核 MsRE 与 LoRA 在 CloudSEN 上的“适配分支开启”和“已知域切换关闭”条件，解释关闭 MsRE 后相对旧 source-only 的 `0.04` 差异；随后完成 source-only 与 MsRE 的逐 scene 配对 bootstrap、leave-one-scene-out，并以同一个 seed64 源 checkpoint 补做目标适配 seed65/66。

## 改动与运行

- 为 LoRA 增加不合并权重的严格低秩分支开关；关闭时逐层只执行冻结 base linear。
- 新增统一 fp16、512×512、三父类推理审计，逐张量核对共享参数与缓冲区，并记录预测 SHA256。
- 新增逐 scene 配对混淆矩阵、10,000 次单 seed scene bootstrap、leave-one-scene-out，以及 20,000 次 seed×scene 分层 bootstrap。
- L8 与 SPARCS 各补 seed65/66，共 4 个作业；RTX 4090 D 串行运行，每个 4000 steps，均使用同一 seed64 source-parent checkpoint。四个作业退出码均为 0，各 80 个 loss 记录全部有限，无 traceback 或 OOM。
- 首次推理因 `FrozenBackboneEncoderDecoder.train()` 未返回 `self` 而在模型加载阶段退出，未产生指标；修复后保留失败目录并重跑。Git push / pull 正常；本轮未下载数据，也未读取 target-test。

## 源域开关审计

- 旧四类 checkpoint 聚合三父类：CloudSEN mIoU `77.8538`；另训三父类 source-parent checkpoint：`77.8179`。旧 `0.04` 是不同 checkpoint / readout 的混合比较，不是 MsRE 关闭后的遗忘。
- L8-MsRE、SPARCS-MsRE、L8-LoRA、SPARCS-LoRA 关闭增量分支后，均与三父类 source-parent 的预测 SHA256 完全一致；共享参数和缓冲区 mismatch 均为 0。
- 增量分支开启时，CloudSEN mIoU 分别为：L8-MsRE `73.3432`、SPARCS-MsRE `73.7240`、L8-LoRA `66.6042`、SPARCS-LoRA `68.1511`。因此只支持“已知域条件下可切换适配”，不支持“开启同一适配模型兼顾源域与目标域”。

## 多 seed 真实结果

固定 source-only：L8 mIoU `58.3357`、Shadow `27.1515`；SPARCS mIoU `57.7128`、Shadow `22.0322`。

| 域 | seed | MsRE mIoU | ΔmIoU | MsRE Shadow | ΔShadow |
|---|---:|---:|---:|---:|---:|
| L8 | 64 | 61.0182 | +2.6825 | 26.7668 | -0.3847 |
| L8 | 65 | 61.9929 | +3.6572 | 28.9796 | +1.8281 |
| L8 | 66 | 61.5397 | +3.2040 | 28.5583 | +1.4067 |
| SPARCS | 64 | 67.1705 | +9.4577 | 41.6433 | +19.6111 |
| SPARCS | 65 | 67.9511 | +10.2383 | 42.2676 | +20.2353 |
| SPARCS | 66 | 67.1013 | +9.3885 | 42.5881 | +20.5558 |

- L8：跨 seed 平均 ΔmIoU `+3.1813 ± 0.4877`（样本标准差），平均 ΔShadow `+0.9501 ± 1.1750`。seed×scene bootstrap 的 ΔmIoU 95% CI 为 `[-0.2701,+6.6341]`，ΔShadow 为 `[-3.1690,+6.1494]`；跨 seed 平均的 leave-one-scene-out ΔmIoU 全为正（`+2.5067` 至 `+5.2832`），但 ΔShadow 并非全为正。
- SPARCS：跨 seed 平均 ΔmIoU `+9.6948 ± 0.4719`，平均 ΔShadow `+20.1341 ± 0.4805`。seed×scene bootstrap 的 ΔmIoU 95% CI 为 `[+2.0908,+15.2728]`，ΔShadow 为 `[+4.9748,+31.7497]`；leave-one-scene-out 两项均全部为正。

## 止损判读

- 目标适配的 seed 随机性较小，SPARCS 的 mIoU 与 Shadow 收益同时稳定。
- L8 的 mIoU 点估计在三个 seed 和所有 leave-one-scene-out 中均为正，但 seed×scene CI 仍跨零；Shadow 在 seed64 为负，分层 CI 也跨零。L8 不能宣称稳定改善弱类。
- 两个目标域的证据方向不一致，因此当前 RGB 结果仍不足以支持“一般性的稳定跨域收益”或进入 64C。可保留的窄主张是：MsRE 在已知域切换条件下精确恢复源路径，并在 SPARCS 上取得稳定的目标收益；L8 仅为有重复性的 mIoU 正点估计，scene 不确定性尚未排除。
- target-test 读取为 0；三个目标适配 seed 共用一个源 checkpoint，本实验只检验目标适配随机性，不代表源训练随机性。

## 产物哈希

- `source_switch.json`: `44cb482b64dc4748e3d3dbcd9b6e517b07e8a0b3b3aa352e6e6cb769ea327576`
- `paired_scene.json`（seed64）: `4929cbe23d6f712d572e298e646e60334fff034f88e9d4a5d86714ea7f2962bc`
- `multiseed_summary.json`: `a2dded2d1e24578ce3f296c8f432175aa49d0cfa7eadd7b5ed6e6798982517f6`
