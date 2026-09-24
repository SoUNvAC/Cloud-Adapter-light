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
