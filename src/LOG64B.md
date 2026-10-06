# Phase 64B 实验记录

## 目标

只改变输入光谱配置，检验 RGB 条件下的 MsRE 迁移—保持优势能否在 Sentinel-2 / Landsat 公共六波段下复现。先完成六波段输入协议审计并重训 CloudSEN 源模型；只有 source-val 三类 mIoU 相对 RGB 源模型下降不超过 1 点，才启动 L8 与 SPARCS 的 source-only / MsRE 目标域实验。

## 改动与协议

- 固定六波段顺序：Sentinel-2 `B4/B3/B2/B8/B11/B12`；Landsat-8 `B4/B3/B2/B5/B6/B7`。
- Sentinel-2 反射率按官方缩放读取，连续波段使用 area 重采样到 20 m，标签使用 nearest；Landsat-8 与 SPARCS 保持原 30 m 网格，不宣称两个传感器像元物理尺度一致。
- nodata / 非有限值转零并生成有效性掩膜；饱和值按协议裁剪。归一化统计仅由 CloudSEN train split 计算：mean=`[0.20777850,0.20358557,0.21717624,0.29672984,0.22447699,0.15877002]`，std=`[0.20317571,0.18181795,0.18963913,0.17799714,0.14027748,0.11965738]`。
- 六通道 patch embedding、adapter depthwise 与 pointwise 输入层均显式扩展初始化；未把 RGB checkpoint 伪装为六通道模型。源训练仅更新六波段 stem、原有 CloudAdapter 与同一 readout，DINO blocks 冻结；未增加 head、loss 或 router。
- 输入审计摘要：`src/work_dirs/phase64b_input_protocol/summary.json`，SHA256 `744d9512939bc9c1384a5c8d51c893a3d8397b01180d333b5c12136d78c2afc1`。L8 manifest SHA256 `885c2d7f61cae23409a7b1ccbe65739c302fc1e59b089aab0cc05badf674872b`；SPARCS manifest SHA256 `7fd78c026d7ee76e3fc8e3dc6f775ddeb86312242650d6ea085502622d05738c`。
- RGB 源 checkpoint SHA256 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`；六通道初始化 checkpoint SHA256 `774b15718b8f211bef361d50c7f24041f283a1f2aaf17a6a30201789de9c97bb`。
- target-test 读取次数为 0；CloudSEN test 数据未下载。L8 与 SPARCS target-train 六通道读取预检通过，但未启动目标域训练。

## 网络与运行情况

- 官方 CloudSEN train / val 六波段约 27 GB 下载完成。首次下载发现 Hugging Face Xet 默认写入用户级缓存后立即终止；随后将全部缓存固定在项目目录内，未删除授权目录外缓存。
- 两次训练前配置错误分别来自配置中暴露 `Path` 类和不可 deepcopy 的文件句柄；均仅在本地修复、验证、push，远端 pull 后重启。失败目录保留用于审计。
- 正式运行使用远端 `conda activate cloud-lite-pt210`，4000 steps，80 个 loss 观测全部有限，范围 `8.4014–20.6970`，无 traceback；完成 8 次验证并生成 `TRAIN_COMPLETE`。

## 止损线

- 64B-1：CloudSEN 六波段 source-val 三类 mIoU 相对 RGB 源模型下降不得超过 1.0 点。
- 只有通过 64B-1 才运行 L8 / SPARCS 的 six-band source-only 与 MsRE；因此目标域增益、Shadow 门和 PEFT 参数门在本轮均未进入判定。

## 真实结果与结论

- RGB 源模型参考 mIoU：`77.8538`。
- stems-only 诊断运行 mIoU：`61.50`，下降 `16.3538` 点；该运行仅用于暴露训练能力不足，不作为正式六波段结论。
- 修正后的正式六波段源模型在 iter 4000 达到最佳 mIoU `58.73`：Surface `74.06`、Cloud `65.49`、Shadow `36.64`；相对 RGB 源模型下降 `19.1238` 点。
- 最佳 checkpoint：`src/work_dirs/phase64b_source_parent/seed64/best_mIoU_iter_4000.pth`，SHA256 `e12714f732e50ec9f2945cc4ec0b8b763fe15992275dab48f2997d65b6727c40`。
- **判定：64B-1 明确失败，按预注册规则停止。** 未运行 L8 / SPARCS 六波段 source-only 或 MsRE，因而没有目标域结果，也不能声称六波段复现了 MsRE 稳健性或带来额外收益。
- 限制：本协议同时把 Sentinel-2 输入落实为原始反射率的统一 20 m 管线，而旧 RGB 参考来自既有 RGB 管线。当前失败说明这套预注册六波段输入配置未保住源性能；若要把损失单独归因于“增加波段”，必须另做同一 20 m / 同一辐射处理下的匹配 RGB 控制，不能用本结果直接归因。
