# Phase 63D：第二独立目标域与最强可复现基线方案

> 状态：仅完成资料、协议与可复现性审计；未下载数据、未训练、未运行远端实验。  
> 资料访问日：2026-09-28。  
> 本文中的“计划命令”均为下一阶段执行规格，不表示本轮已经执行或获得结果。

## 1. 结论先行

**推荐第二目标域：Deep Fmask Dataset 的 Sentinel-2 雪冰/高山域。**

- 数据集包含 45 个 Sentinel-2 L1C scene，人工标签为 `clear-sky land / cloud / shadow / snow / water`，官方划分为 22 个 validation scene 和 23 个 test scene。
- 它与 Landsat-8 Biome 在传感器、影像、标注者和标签协议上独立；与 CloudSEN12 在影像集合和标注流程上独立，但同为 Sentinel-2 MSI，**不是传感器独立域**。正式使用前仍须按完整 SAFE product ID、MGRS tile 和空间足迹核验是否与 CloudSEN12 scene 重叠。
- 该域的主要价值是把 Phase 63 的“可辨识性”命题置于最危险的 bright-surface 条件：snow/ice 与 cloud 的混淆，以及 terrain shadow 与 cloud shadow 的混淆。snow/ice 必须作为原生类和误差分层完整报告。
- 该数据没有 thin/thick 像元标签。因此它只能验证：
  1. 三类 `clear-surface / cloud / shadow`；
  2. 二类 `cloud / non-cloud`；
  3. 可选 `usable / contaminated(cloud+shadow)`；
  4. snow/ice、water、clear-land 条件稳定性。  
  **不得报告第二域四类 mIoU，不得把 native cloud 拆成 thin/thick，也不得把 snow/water 偷并为“clear”后隐藏其失败。**
- 同一测试集上公开报告最强且有官方代码的基线是论文的 **Deep F-Mask Stage-4 self-training U-Net**：论文报告 test mIoU `0.82`，Cloud/Shadow/Snow IoU 分别为 `0.93/0.68/0.87`。这只是文献锚点，尚未在本仓库复现。
- 由于作者未发布预训练权重，且 GitHub 仓库未显示明确软件许可证，1–2 天内不应承诺完整复训。短期最强可执行方案是冻结本项目模型并并行运行 Fmask、Sen2Cor、OmniCloudMask 等公开推理基线；Deep F-Mask Stage-4 作为后续严格复现任务，过不了许可、数据或复现门即降级为“文献参考”，不能冒充已复现基线。

## 2. 本地仓库盘点

### 2.1 已有数据与协议

| 本地对象 | 传感器/输入 | 本地标签 | 当前状态 | 对 Phase 63D 的意义 |
|---|---|---|---|---|
| `dataset/cloudsen12_high.py`、`cloud_adapter/datasets/cloudsen12_high_l1c.py` | Sentinel-2 L1C；loader 默认 RGB，接口可选更多波段 | clear、thick、thin、shadow | 当前源域与 Phase 63 主域 | 不能同时充当第二目标域 |
| `dataset/l8_biome.py`、`cloud_adapter/datasets/l8_biome.py` | Landsat-8 OLI/TIRS；代码列出 B1–B11，默认 RGB | clear、shadow、thin、cloud | 已被 Phase 28、45–62 反复使用 | 已是第一目标域，不再算“第二独立域” |
| `dataset/hrc_whu.py`、`configs/_base_/datasets/hrc_whu.py` | RGB 高分辨率影像 | clear/cloud 二类 | 仓库已有配置和历史结果 | 不能验证 shadow 或细粒度可辨识性 |
| `dataset/gf12ms_whu.py`、GF1/GF2 配置 | GF-1/GF-2；代码支持 RGB+NIR，历史 mmseg 数据为二类 | clear/cloud 二类 | 仓库已有配置和历史结果 | 可作二类外推附录，不足以作为主第二域 |

本地历史 `eval_result` 中 HRC/GF1/GF2 的结果来自各自域模型及各自 test/val 配置，并非 CloudSEN12→第二域的冻结迁移结果，不能直接作为 Phase 63D 基线。Phase 45 也已明确规定 HRC→GF 的二类证据不得与 L8 四类指标直接平均。

### 2.2 本地共同波段协议

延续 Phase 60 的 S0–S4 定义；第二域不得另选“更有利”的波段：

| 条件 | Sentinel-2 | Landsat-8 | 说明 |
|---|---|---|---|
| S0 RGB | B4, B3, B2 | B4, B3, B2 | 三通道共同输入 |
| S1 +NIR | +B8（若统一到 20 m，也要固定 B8/B8A 选择） | +B5 | B8 与 B8A 不得按结果事后切换 |
| S2 +SWIR1 | +B11 | +B6 | 公共光谱语义 |
| S3 +SWIR2 | +B12 | +B7 | 公共光谱语义 |
| S4 indices | NDVI、NDMI、NDSI、NBR | 同公式 | 只由 S3 六通道确定性计算 |
| S5 sensor-all | Sentinel-2 全部可用 L1C 波段 | Landsat B1–B11 | 仅传感器上界，不得与公共波段主结果混称 |

Deep Fmask 原论文使用 Sentinel-2 MSI L1C TOA 数据，并在 20 m 尺度处理标签/输出。复现时需先从作者代码生成的数据 shape 和通道列表核验实际输入；未完成该审计前，不把“13 个 MSI 波段”写成“网络实际使用了 13 个通道”。Phase 63D 主比较必须至少报告 S0，并优先报告与 Phase 60 一致的 S3；S5 只能作为附加上界。

## 3. 候选域比较

| 候选 | 原生标签与规模 | 独立性 | 优点 | 决定 |
|---|---|---|---|---|
| **Deep Fmask Dataset（推荐）** | 45 scenes；clear-land/cloud/shadow/snow/water；22 val + 23 test | 独立人工标注；相对 L8 传感器独立；相对 CloudSEN 同传感器但数据/协议独立，待 scene 交叉核验 | 冻结 scene 列表、雪冰高风险域、shadow 和 bright-surface 都可评估 | **主第二域** |
| Sentinel-2 Cloud Mask Catalogue（S2-CMC） | 513 个 1022×1022、20 m 子场景；像素三类 clear/cloud/cloud-shadow；424 个可评 shadow，89 个 shadow 边界不可标 | 2018 S2 随机样本，独立于 L8；与 CloudSEN 同传感器；后续 GDSD/CloudS2Mask 使用过它 | 更大、带 surface/cloud thickness 场景 tag | 备选；89 个 partial-shadow 必须部分标签处理，且公开记录页面未明确展示许可证，暂不做主域 |
| GDSD | CESBIO、S2-CMC 与 SDSU-MSU 三来源；三类 fill/shadow/clear/cloud；88.7 GB | 不是单一新独立语料，明确派生/精修自已有公开集 | 大规模且有官方 Swin-Unet 推理代码 | 不做锁定第二域；来源混合、无清晰冻结 test protocol、成本高 |
| Landsat-8 SPARCS | 80 个 1000×1000 子场景；shadow、shadow-over-water、water、snow、land、cloud、flooded | 标注项目独立，但与 L8 Biome 同为 Landsat-8 Pre-Collection；单一分析员 | 官方 USGS、标签丰富 | 作为同传感器外部附录；不够“第二传感器/第二观测域” |
| HRC-WHU / GF1MS-WHU / GF2MS-WHU | clear/cloud 二类 | 传感器可独立 | 仓库已有 loader/config | 只能验证粗粒度 cloud detection，不能支撑选择性细粒度主张 |

推荐 Deep Fmask 而非 S2-CMC 的关键原因不是类别更多，而是它有冻结的 scene 名单、独立人工标注和明确 snow/ice 压力条件。S2-CMC 的 `CLOUD THICKNESS=thin/thick` 是**子场景 tag**，不是每像元 thin/thick mask，不能升级成四类真值。

## 4. 推荐域的数据、许可和获取

### 4.1 数据实体

- 人工标签：PANGAEA 数据集，DOI `10.1594/PANGAEA.942321`，总发布包约 2.1 MB；包括标签映射、22/23 个 scene 的 SAFE 文件名清单和对应 GeoTIFF 标签。
- 原始影像：清单指向 Sentinel-2A/B L1C `.SAFE` 产品，采集时间为 2019-10 至 2020-12；影像不随标签包分发，需要按完整产品名从 Copernicus Data Space Ecosystem 获取。
- 论文训练域：另有 96 个训练 scenes，以 Fmask 伪标签进行四阶段 self-training；人工 validation/test 标签不用于训练。Phase 63D 的短期零训练审计不需要这 96 个 scenes，完整 Deep F-Mask 复现才需要。
- 当前作者 GitHub 的 `exp_data/validation_filename.txt` 页面显示 20 行产品 ID，而论文和 PANGAEA 均声明 22 个 validation scenes。这是已发现的协议差异，不得默认为等价：评价 scene 以 PANGAEA 的 22/23 标签包为权威；复训前逐项比较 GitHub 与 PANGAEA 清单并把差集写入审计表。

### 4.2 许可边界

| 对象 | 许可/条款 | 使用要求 |
|---|---|---|
| PANGAEA 人工标签 | CC BY 4.0 | 保留作者、数据集、DOI 与许可证归属；衍生 manifest/统计需注明来源 |
| Sentinel-2 L1C | Copernicus Sentinel free, full and open access | 获取通常需注册；发布或分发时按官方 legal notice 标注 `Copernicus Sentinel data [Year]`；修改数据时注明 `Contains modified Copernicus Sentinel data [Year]` |
| Deep F-Mask GitHub 代码 | 仓库公开，但截至访问日未见 LICENSE 文件或 README 中的明确软件许可 | 可以审阅公开方案；在许可澄清前不把代码复制进本仓库、不发布修改版，也不把运行权利解释为开源授权 |
| 论文 | Remote Sensing 开放论文 | 正常引用 DOI；论文结果只能称“reported”，复现前不能称“our reproduced baseline” |

获取入口：

1. [Deep Fmask Dataset，PANGAEA](https://doi.org/10.1594/PANGAEA.942321)（访问：2026-09-28）。
2. [Copernicus Data Space Ecosystem：Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2)（访问：2026-09-28）。
3. [CDSE STAC 官方文档](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)（访问：2026-09-28）。
4. [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)（访问：2026-09-28）。

## 5. 标签粒度与评价映射

### 5.1 保留的原生评价

先按 Deep Fmask 原生五类报告，不丢类别：

```text
1 clear-sky land
2 cloud
3 cloud shadow
4 snow/ice
5 water
0 no-data / ignore
```

必须报告每类 IoU、F1、precision、recall、scene-macro 指标和 scene bootstrap 95% CI。`no-data` 只作 ignore，不进入均值。

### 5.2 与本项目输出的合法公共映射

| CloudSEN/L8 预测 | Deep Fmask 三类 | 二类 cloud/non-cloud | usable/contaminated |
|---|---|---|---|
| thick cloud | cloud | cloud | contaminated |
| thin cloud | cloud | cloud | contaminated |
| cloud shadow | shadow | non-cloud | contaminated |
| clear | clear-surface | non-cloud | usable |

Deep Fmask 真值侧：

- `cloud → cloud`；
- `shadow → shadow`；
- `clear-land + snow + water → clear-surface`，但三者的条件误差必须另表；
- 二类中只有 native cloud 为 cloud，其余有效类为 non-cloud；
- usable/contaminated 中 native cloud+shadow 为 contaminated，其余有效地表类为 usable。

这张映射不创造 thin/thick 真值。对 GT snow/ice 至少额外报告：预测为 cloud、shadow、clear 的比例；对 GT water 同样报告。若模型把 snow 全部判为 clear，三类总体分数可能很好，但原生五类和 snow 条件表会如实暴露信息损失。

### 5.3 Phase 63 选择性指标在第二域的解释

- 风险标签：冻结模型在三类映射上是否预测错误；原生五类错误另报，不能混为一个 AUROC。
- fine prediction：`clear-surface/cloud/shadow` 三类；coarse prediction：`cloud/non-cloud` 二类。
- coverage-risk：按像元计算并以 scene 为 bootstrap 单位；同时报告每 scene 覆盖率分布，避免大 scene 支配。
- 类别覆盖门：第二域没有 thin，故不能声称复现 Thin≥30% 的门。只检查 native cloud 和 shadow 各自 fine coverage，snow/ice 另报拒绝率。
- Phase 63 的 router/risk calibrator 在 CloudSEN/L8 development 上冻结后，第二域只允许一次前向评估；不得用这 22/23 个 scene 调阈值后再称外部验证。

## 6. Scene 独立划分与无泄漏核验

### 6.1 冻结划分

- `target-dev`：官方 22 个 validation scenes，仅用于第二域数据/映射调试和风险方向检查。
- `target-confirm`：官方 23 个 test scenes，manifest hash 锁定后封存，只做一次最终评价。
- `target-unlabeled-train`：仅在复现 Deep F-Mask 时使用官方 96 个 training scenes；不得读取 validation/test 标签参与优化。
- 为复现论文，保留一份完全不改动的 `official_split` 结果；为 Phase 63 主文，再生成通过更严格 tile/footprint 检查的 `strict_independent_split`。二者不得混算。

### 6.2 分组键

最低要求不是 patch 文件名，而是以下层级：

1. 完整 Sentinel SAFE product/granule ID；
2. MGRS tile（产品名中的 `Txxxxx`）；
3. acquisition timestamp；
4. raster footprint 的空间相交关系。

推荐 `scene_group = MGRS tile`。同一 MGRS tile 的多时相影像不得跨 `target-dev` 与 `target-confirm`。如果官方 22/23 划分存在 tile 重叠：

- `official_split` 仅用于复现作者数字，并明确不是严格位置独立；
- Phase 63 主结果把重叠 tile 全部保留在 confirm 或整体隔离，绝不将 confirm scene 移入开发；
- 若处理后 confirm 少于 15 个独立 tile，或 shadow/cloud 任一类只存在于少于 5 个 confirm scenes，则停止把该域作为定量主证据，只保留压力测试。

### 6.3 对现有两域的交叉核验

必须生成以下三张交集表并存 SHA256：

```text
deepfmask_vs_cloudsen12_exact_product.csv
deepfmask_vs_cloudsen12_tile_time_footprint.csv
deepfmask_vs_l8biome_footprint.csv
```

- CloudSEN12：由于同为 Sentinel-2，检查 exact SAFE ID、MGRS tile、时间与 footprint；任何 exact product 重叠 scene 从第二域剔除。仅地理重叠但时相不同须报告，不能称“地理独立”。
- L8 Biome：传感器和产品 ID 天然不同，但仍检查 footprint；空间重叠不等于数据泄漏，不过会削弱“地理独立”表述。
- Deep Fmask 标注没有从 CloudSEN12 或 L8 Biome 派生；S2-CMC/GDSD 则存在明确的派生关系，因此未被选为主域。

## 7. 最强公开基线与复现层级

### 7.1 基线优先级

| 级别 | 基线 | 用途 | 可复现性结论 |
|---|---|---|---|
| B0 | 本项目冻结 source-only 与当前最强 adapted checkpoint | 回答现有模型是否跨到雪冰域；不在第二域选 checkpoint | 仓库内可复现，需新增 dataset adapter 后才能运行 |
| B1 | Fmask 4、Sen2Cor 2.8 | 与原论文完全同版本的物理/业务基线 | 官方工具可获得，但旧版本环境需容器化 |
| B2 | Deep F-Mask Stage-4 | 同一测试集公开报告最强基线；test mIoU 0.82 | 有作者代码和命令、无公开权重；完整训练昂贵；软件许可未明确 |
| B3 | OmniCloudMask 冻结公开权重 | 当代强深度基线；RGB+NIR 类输入，输出 clear/thick/thin/shadow | 安装和推理方便，但训练数据包含 CloudSEN12，且未在该 45-scene test 上给出公开成绩；只能称 challenger |
| B4 | Fmask 5 UPL | 当前公开、MIT 许可的 physics-informed ML 基线 | 命令明确，但不等同论文中的 Fmask 4，结果必须另列 |

“最强”严格指 Deep F-Mask 论文在**同一 Deep Fmask test set** 上报告的对比：Stage-4 mIoU 0.82；Fmask 4 为 0.57；Sen2Cor 2.8 为 0.53。不能用不同数据集上的 CloudS2Mask/OmniCloudMask 数字宣称更强。

### 7.2 文献复现目标

Deep F-Mask 原论文 test 指标锚点：

| 类别 | IoU |
|---|---:|
| clear-sky land | 0.85 |
| cloud | 0.93 |
| shadow | 0.68 |
| snow | 0.87 |
| water | 0.77 |
| mIoU | 0.82 |
| total accuracy | 0.93 |

这些数值来自论文 Table 9，仅用于复现验收，不是本项目实验结果。

### 7.3 命令级计划

以下命令仅定义后续操作顺序；必须在独立外部工作目录执行，不在本轮运行。

#### A. 获取并锁定官方清单

```bash
# 标签包：从 PANGAEA DOI 页面人工确认 CC BY 4.0 后获取。
# 原始 L1C：按 test/validation SAFE product name 通过 CDSE STAC/OData 解析。
python tools/phase63d_build_manifest.py \
  --pangaea-root /data/deep_fmask_labels \
  --cloudsen-manifest work_dirs/phase63/frozen_cloudsen_manifest.csv \
  --l8-manifest work_dirs/phase45_protocol_audit/locked_manifest.csv \
  --group-key mgrs_tile \
  --out work_dirs/phase63d/manifests

python tools/phase63d_audit_manifest.py \
  --manifest work_dirs/phase63d/manifests/strict_independent.csv \
  --require-scene-disjoint \
  --require-zero-exact-cloudsen-overlap \
  --seal-confirm
```

上述两个 `phase63d_*` 脚本当前不存在；这是下一阶段应实现的固定 CLI，不能在论文中写成已经可运行。

#### B. 原论文 Deep F-Mask 完整复现

```bash
git clone https://github.com/kmlnbr/deep-fmask.git external/deep-fmask
cd external/deep-fmask
git rev-parse HEAD > ../deep_fmask_commit.txt
conda env create --name DFMask --file environment.yml
conda activate DFMask

# 将官方列出的 TRAIN/VALIDATION/TEST SAFE 产品放入 exp_data 对应目录；
# 为训练 scenes 运行 Fmask 4，并按 README 生成 *FMASK.tif。
cd src
python make_network_data.py --mode train
python make_network_data.py --mode test
./pipeline.sh phase63d_deepfmask
python predict.py -e phase63d_deepfmask_stage3 \
  -p /data/deep_fmask/TEST
```

注意：作者 README 把最后模型命名为 `stage3`，论文称四阶段；报告时写“Stage-4 / code stage3”，避免 off-by-one。训练前必须记录代码 commit、环境导出、Fmask 版本、所有 scene/label hash。若未获得软件许可澄清，代码只能在隔离环境中用于内部复现，不得复制进本仓库或再分发。

#### C. 当前 Fmask 5 可执行基线

```bash
git clone https://github.com/GERSL/Fmask.git external/fmask
cd external/fmask
# 禁止直接使用移动的 develop HEAD；先将完整 5.0.1 package 与源码历史对应，
# checkout 到经核验的 5.0.1 commit，并记录 package/auxiliary-model SHA256。
git checkout <VERIFIED_FMASK_5_0_1_COMMIT>
git rev-parse HEAD > ../fmask_commit.txt
conda create -n fmask python=3.10
conda activate fmask
# 按官方 README 安装依赖并取得完整 package/auxiliary model。
cd main
python fmask.py \
  --imagepath /data/deep_fmask/TEST/<SCENE>.SAFE \
  --model UPL --dcloud 0 --dshadow 0 --dsnow 0 \
  --output /outputs/phase63d/fmask5/<SCENE>
```

必须显式将 dilation 固定为 0，避免后处理半径成为隐式调参。Fmask 5 和论文 Fmask 4 分开列。

#### D. OmniCloudMask 冻结 challenger

```bash
python -m venv /envs/omnicloudmask
source /envs/omnicloudmask/bin/activate
pip install omnicloudmask
pip freeze > /outputs/phase63d/omnicloudmask_requirements.txt

python tools/phase63d_run_omnicloudmask.py \
  --manifest work_dirs/phase63d/manifests/strict_independent.csv \
  --bands red green nir \
  --model-version frozen-at-install \
  --out /outputs/phase63d/omnicloudmask
```

正式运行时必须把包版本、两个 ensemble 权重的 SHA256 与下载来源写入 manifest；`phase63d_run_omnicloudmask.py` 同样是待实现接口。其四类输出只在映射到三类后与 Deep Fmask GT 比较。

#### E. 本项目冻结 checkpoint

```bash
cd src
python tools/test.py \
  configs/protocol/phase63d_deepfmask_s0.py \
  <FROZEN_CHECKPOINT> \
  --work-dir work_dirs/phase63d/<MODEL>/official_split

python tools/phase63d_evaluate_selective.py \
  --predictions work_dirs/phase63d/<MODEL>/official_split/predictions \
  --manifest work_dirs/phase63d/manifests/official_split.csv \
  --native-labels clear_land cloud shadow snow water \
  --mapped-levels native5 three_class binary valid_invalid \
  --bootstrap-unit scene --bootstrap-draws 2000 \
  --out work_dirs/phase63d/<MODEL>/metrics.json
```

这些项目内 config/evaluator 尚未创建；D 组本轮不改代码。后续实现必须固定 checkpoint，不得根据第二域 confirm 表现选择模型、阈值、TTA 或波段。

## 8. 算力、存储和时间预算

以下是工程估算，必须在首个 scene 上实测后更新，不是已观测数据。

| 工作 | 资源估算 | 时间上限 | 说明 |
|---|---:|---:|---|
| 标签、scene 清单、许可与 overlap 审计 | CPU，<2 GB 标签/元数据工作盘 | 4–8 h | 不含 L1C 下载 |
| 45 个 eval SAFE 获取与校验 | 约 25–60 GB 下载量（待 CDSE 实测） | 4–12 h | 受网络与 CDSE 配额影响 |
| 45 scenes 的冻结项目模型 + Omni 推理 | 1×RTX 4090D，约 50–150 GB 缓存/输出 | 2–8 h | 取决于切片、S0/S3 和 TTA；主结果禁用未预注册 TTA |
| Fmask/Sen2Cor 全 test | 16–32 CPU cores | 6–18 h | Fmask 5 官方示例约 11.9 min/scene，仅作量级参考 |
| Deep F-Mask 四阶段完整复训 | 1×RTX 4090D；96 train + 22 val scenes | 先做 1% throughput 实测；硬上限 72 GPU-h | 论文每阶段 100 epochs；若外推超过 72 GPU-h，本阶段不复训 |
| 全量 141 SAFE + 解压/patch/cache | 预留 0.5–1.0 TB 工作盘 | 数据到齐后再核算 | 禁止因空间不足删除原始 hash/manifest |

**1–2 天可交付范围**：完成 manifest/许可/重叠审计，取得并校验 45 个 eval scenes，运行不训练的冻结模型与至少一个公开推理基线，输出三类/二类/native snow 分层结果。完整 Deep F-Mask Stage-4 复训不纳入 1–2 天承诺。

## 9. 预注册失败条件

满足任一项即停止把 Deep Fmask 写成“独立第二目标域主证据”：

1. 无法取得全部 23 个 locked test scenes 或其标签/影像 hash 不匹配；
2. 与 CloudSEN12 存在 exact SAFE product 重叠且无法无损剔除；
3. 严格 MGRS-tile 分组后 confirm 少于 15 个独立 tile，或 cloud/shadow 任一类只在少于 5 个 confirm scenes 出现；
4. 原生标签值、no-data 或 20 m 配准无法从官方文件唯一复原；
5. 只能通过把 snow/water 并入 clear 且不单报条件误差才能得到正面结果；
6. 有人把 native cloud 当成 thick，或从 scene-level cloud-thickness tag 伪造 pixel-level thin/thick；
7. confirm 用于调风险阈值、TTA、checkpoint、波段或映射；
8. Deep F-Mask 软件许可仍不明确却需要复制/修改/分发代码；此时只保留论文数字和可合法运行的 Fmask/Omni 基线；
9. 完整复现的官方 split mIoU 与论文 0.82 相差超过 0.02，且无法由版本、scene、配准或 metric 定义解释；不得继续称“复现”；
10. 第二域上 coarse 指标或 snow/ice 条件性能显著崩溃，却只展示选择性 fine 指标；
11. 风险路由器必须用 confirm 标签重校准才能达到门槛；这说明没有外域泛化，终止第二域主张；
12. 第二域没有 thin GT，却被用于宣称 Thin 选择性门已经跨域复现。

若本候选因数据获取或严格划分失败，后备顺序为：S2-CMC 三类 partial-shadow 协议 → SPARCS 原生多地表类协议。HRC/GF 二类只作附录，不升级为第二目标域。

## 10. 对论文主张的允许范围

若 Deep Fmask 严格 confirm 通过，可写：

> 选择性多粒度策略在独立 Sentinel-2 雪冰/高山标注集上，对三类 cloud-shadow-surface 决策及二类 cloud detection 展现一致方向，并显式报告 snow/ice 条件失败。

不能写：

- “在第二个四类数据集验证了 thin/thick/shadow”；
- “跨第二传感器验证”，因为 Deep Fmask 与 CloudSEN12 都是 Sentinel-2；
- “Deep F-Mask 基线已复现”，除非执行完整官方协议并通过数值验收；
- “所有像元标签完全兼容”，因为 clear-land/snow/water 与 CloudSEN clear 的本体不同，shadow 也可能包含地形阴影。

最稳妥的 TGRS 证据组合是：L8 Biome 提供跨传感器四类主审计；Deep Fmask 提供独立标注流程下的雪冰可观测性和三/二类层级外部验证。两域指标分别报告，不做无意义平均。

## 11. 一手来源

全部链接访问于 2026-09-28：

1. [Nambiar et al., 2022, *A Self-Trained Model for Cloud, Shadow and Snow Detection in Sentinel-2 Images of Snow- and Ice-Covered Regions*](https://doi.org/10.3390/rs14081825)。数据规模、96/22/23 scene、四阶段训练、原生类别与 Table 9 指标来源。
2. [Deep Fmask Dataset，PANGAEA DOI 10.1594/PANGAEA.942321](https://doi.org/10.1594/PANGAEA.942321)。人工标签、22/23 scene 清单、CC BY 4.0 与获取说明。
3. [作者 Deep F-Mask 官方代码](https://github.com/kmlnbr/deep-fmask)。环境、数据准备、`pipeline.sh`、预测命令与未提供预训练权重的核验来源。
4. [Copernicus Sentinel-2 官方数据入口](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2) 与 [CDSE STAC 文档](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)。原始 L1C 获取方式。
5. [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)。Sentinel 数据 free/full/open 条款与署名要求。
6. [Sentinel-2 Cloud Mask Catalogue, Zenodo 4172871](https://doi.org/10.5281/zenodo.4172871)。513 子场景、三类像素 mask、424/89 shadow 状态和 scene-level tag。
7. [GDSD, Zenodo 14397827](https://doi.org/10.5281/zenodo.14397827) 与 [官方 Swin-Unet 推理代码](https://github.com/hankui/Cloud-mask-Sentinel-2)。三来源派生关系、三类标签、88.7 GB 规模和命令。
8. [USGS SPARCS Validation Data](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data)。80 个 Landsat-8 子场景、原生七值标签与单分析员限制。
9. [OmniCloudMask 官方代码](https://github.com/DPIRD-DMA/OmniCloudMask)。MIT 许可、推理接口、模型标签和训练数据版本说明。
10. [GERSL Fmask 官方代码](https://github.com/GERSL/Fmask)。MIT 许可、Fmask 5 UPL 命令、输出标签及已知 snow/thin/shadow 失败模式。
