# Phase 60 实验记录

## 2026-09-24：协议冻结与启动

- 目标：停止新增分割模型，独立审计多光谱信息充分性（60A）与目标标注支持上界（60B）。
- 改动：新增 `tools/run_phase60a_multispectral_audit.py`、`tools/prepare_phase60b_support_audit.py`、`tools/evaluate_phase60b_support_audit.py` 和远程串行入口 `tools/run_phase60_4090d.sh`。
- 网络/读出：60A 不运行分割网络，只使用 Phase 54 已对齐缓存和固定 class-balanced ridge；60B 冻结 Phase 22 source 网络做非 oracle acquisition，冻结 Phase 52 adapted 网络提取 `pixel_decoder_mask`，仅拟合同一个 class-balanced ridge。Oracle 只作上界，禁止进入最终方法。
- 数据口径：只使用 USGS `Shadows?=yes`。合法 target-train 池为 1560 张/18 scenes，target-val 为 963 张/8 scenes；Phase 50 原 65 张中仅 12 张合法。指定预算 16/32/65/130/325 不变，但占合法池比例为约 1.03%/2.05%/4.17%/8.33%/20.83%，不伪称 0.25%–5%。
- 60A 输入：S0 RGB；S1 +NIR；S2 +SWIR1；S3 +SWIR2；S4 六公共通道加预注册 NDVI/NDMI/NDSI/NBR。S0–S4 使用同一批像素（每图每类最多 32）。
- 范围限制：授权目录内只有 RGB/NIR/SWIR1/SWIR2 对齐缓存；Landsat 全 11 波段原始档案位于授权目录外，因此 S5 明确记为 blocked，不生成替代数据或结果。
- 止损线：60A 同一 readout 下 Thin IoU ≥32.31、Shadow IoU ≥20、mIoU 相对 S0 ≥2，且必须由公共波段条件达到；60B 按 65/130/325 的 oracle 阈值判读采样、标注量或观测/标签瓶颈。
- 结果：尚未产生；不得提前填写。

## 2026-09-24：60A 完成

- 目标：判断固定浅层读出下，公共多光谱通道能否同时补足 thin/shadow 并相对 RGB 提升至少 2 mIoU。
- 改动：无训练中改码；按冻结协议完成 S0–S4，S5 因全 11 波段原始档案不在授权目录内保持 blocked。
- 网络/读出：未训练或微调分割网络；12 张合法 Phase 50 target-train 图块、每图每类最多 32 像素、同一个 class-balanced ridge；963 张/8 scenes 合法 target-val。
- 止损线：Thin ≥32.31、Shadow ≥20、mIoU 相对 S0 ≥2，且公共波段条件达到全部三项。
- 结果（dense 32×32 readout 后最近邻回投）：S0 mIoU/Thin/Shadow/H/AUROC=18.853/3.893/0.386/0.703/0.4172；S1=21.957/9.430/0.478/0.910/0.2887；S2=20.932/1.033/2.333/1.432/0.5172；S3=21.527/5.047/3.627/4.221/0.5626；S4=23.783/12.095/7.602/9.336/0.6413。AUROC 为同像素 shadow-vs-thin sampled probe。
- 稳定性：S4 scene-bootstrap 95% CI：mIoU [14.665,29.779]、Thin [1.186,21.200]、Shadow [2.454,11.457]、弱类调和均值 [2.046,11.921]；各 biome 的 Thin/Shadow 波动明显，公共波段提升不稳定。
- 判定：S1–S4 虽均达到相对 S0 的 +2 mIoU，但均未达到 Thin/Shadow 双阈值，60A `stop_multispectral_line`。结果 SHA256：`64987af34d92c3699b2279efa88c51885f91d1ea72fe7f9c3647140c9e5d8e2a`。

## 2026-09-24：60B 完成

- 目标：用完整 target-train 标签模拟主动学习，判断 65 张失败究竟来自采样还是标注量/观测与标签体系。
- 改动：在 1560 张合法池上完成 random、entropy、core-set、uncertainty+diversity、Phase50-policy-rebased、oracle class-balanced、oracle thin-shadow relation；原 Phase 50 单列为 nominal-65/实际合法 12。Oracle 只作上界。
- 网络/读出：冻结 Phase 22 source 网络计算非 oracle acquisition；冻结 Phase 52 adapted `pixel_decoder_mask`；所有预算使用 seed=56、每图每类最多 32 像素和同一个 class-balanced ridge；dense 验证固定 963 张/8 scenes。
- 止损线：65 张 oracle 同时达到 Thin ≥32、Shadow ≥20 才可判采样瓶颈；若仅 130/325 达到则否定“1%足够”；325 仍失败则归因更根本的数据/标签/传感器可观测性问题。
- 结果（mIoU/Thin/Shadow/H）：原 Phase50 nominal-65（合法 12）=43.245/16.406/18.603/17.435；65 张 random=47.081/26.852/27.749/27.293，Phase50-rebased=44.722/23.608/23.576/23.592，entropy=42.275/15.713/17.114/16.384，core-set=45.076/25.734/20.335/22.718，uncertainty+diversity=41.918/17.036/17.246/17.141，oracle class-balanced=43.324/21.101/25.065/22.913，oracle relation=43.042/25.577/23.514/24.502。
- 上界：325 张 oracle class-balanced=46.490/27.021/27.486/27.252；oracle relation=47.705/28.355/27.517/27.930。两者 Shadow 过线但 Thin 仍低于 32；不存在首次达标预算。
- 判定：`fundamental_data_label_or_sensor_observability_problem`。不能宣称 1% 标注足够，也不能把 oracle 纳入最终方法；仅靠更换当前采样策略不足以跨过双弱类门槛。
- 产物：selection SHA256 `674b644c7d4fbe46763ba924ff09f36a54f4544337714bab85e298756524ad3d`；60B summary SHA256 `623edacc8808e2f1e5ec2818bb02e0ae34a975d5b1ab65d34bcb65a1517da89b`；总摘要 SHA256 `706e2d6adf6cd9a2ffe33b33e33f949c6f93f5d2e375b644912552a07d64049c`。
