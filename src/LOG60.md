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
