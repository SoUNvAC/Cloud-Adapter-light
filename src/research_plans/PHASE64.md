# Phase 64：面向异构标签空间的层级一致性参数高效域适配

> 状态：Day 1 数据集、标签树和 72 小时止损协议已冻结；未训练新模型。
> 冻结日期：2026-10-04。
> 机器执行前必须通过 `tools/audit_phase64_dataset_registry.py`。

## 1. 研究命题

不同云数据集共享稳定的粗粒度语义，却不必共享 Thin、Thick、Shadow 或地表类的判定标准。Phase 64 不学习逐像元标签转移矩阵，也不猜测含糊像元真值，而是把每个数据集公开的原生类别放进一棵预先声明的层级树：

```text
冻结视觉骨干（full fine-tuning 基线除外）
          │
MsRE / LoRA / 其他轻量适配器
          │
共享父类 Head
surface_visible / cloud / shadow
          │
数据集特定条件子类 Head
CloudSEN: clear | thin/thick | cloud-shadow
L8 Biome: clear | thin/cloud | cloud-shadow
Deep Fmask: clear-land/snow/water | cloud | shadow
SPARCS: land/water/snow/flooded | cloud | shadow/shadow-water
```

父类第一项不能简单写成字面意义的 `clear-sky land`。Deep Fmask 和 SPARCS 含有 snow、water、flooded 等原生地表类；把它们静默并入 clear-land 会伪造标签兼容性。因此代码名冻结为 `surface_visible`，论文展示名可写 `Clear-surface`，含义是“有效、未被 cloud 或 shadow 标注的可见地表”。每个原生地表子类仍须单独报告。

## 2. Day 1 数据集表

完整机器可读表位于 `research_plans/protocol_data/phase64_dataset_registry.json`。

| 数据域 | 传感器、波段与分辨率 | 官方原生类别 | Shadow / Thin-Thick | 划分协议 | 父类映射 | 许可 | 当前可执行性 |
|---|---|---|---|---|---|---|---|
| CloudSEN12 High（源） | Sentinel-2 MSI；13 个光学波段，原生 10/20/60 m；Phase 64 固定 L1C | clear、thick、thin、cloud shadow | 都有，像元级 | 仓库 high-quality：8490/535/975 patch；跨数据集前另查 scene/tile 重叠 | clear→surface；thin+thick→cloud；shadow→shadow | CC BY-NC-SA 4.0 | loader 已有，执行机数据待核验 |
| L8 Biome（目标1） | Landsat-8 OLI/TIRS B1–B11；公共光学主协议按 30 m | 0 Fill、64 Shadow、128 Clear、192 Thin、255 Cloud | 有 shadow，但完整性由 scene 的 `Shadows?` 声明 | 官方是 96-scene validation collection，没有官方 train/val/test；使用 Phase 45 已冻结的 biome-stratified scene split | Fill ignore；Clear→surface；Thin+Cloud→cloud；Shadow→shadow | USGS data release；按 USGS 数据政策与产品元数据处理 | loader 已有；三父类完整监督/评价只用 `Shadows?=yes` |
| Deep Fmask（目标2） | Sentinel-2 L1C；主协议 RGB+NIR+SWIR1+SWIR2，统一 20 m | 0 Unlabelled、1 Clear-Sky Land、2 Cloud、3 Shadow、4 Snow、5 Water | 有 shadow；无 thin/thick | 官方 22 validation + 23 test scene，无人工训练 split；若监督适配，把 22 scene 明确改称 `target_adapt`，所有选择在 test 前冻结 | clear/snow/water→surface；cloud→cloud；shadow→shadow；三种 surface 子类分别报告 | 标签 CC BY 4.0；原始影像遵循 Copernicus 条款 | 官方标签包、类别与 22/23 清单已核验；SAFE 原始影像和 loader 未就绪 |
| SPARCS（备份） | Landsat-8 Pre-Collection；80 个 1000×1000 subset，公共协议按 30 m | shadow、shadow-over-water、water、snow、land、cloud、flooded | 有两个 shadow 子类；无 thin/thick | 官方 validation collection，无 train/val/test；若使用必须先按源 scene 预注册拆分 | land/water/snow/flooded→surface；cloud→cloud；两种 shadow→shadow | USGS data release；按 USGS 数据政策与产品元数据处理 | 无 loader、未下载；只作备份或外部附录 |

官方依据：CloudSEN12 项目页说明数据内容和 [CC BY-NC-SA 4.0](https://cloudsen12.github.io/Legacy/)；USGS 的 [L8 Biome 发布页](https://landsat.usgs.gov/node/7) 给出 96 scenes、八个 biome、raw mask ID 和 `Shadows?` 字段；[Deep Fmask PANGAEA 发布页](https://doi.org/10.1594/PANGAEA.942321) 给出五个有效类、22/23 scene 与 CC BY 4.0；[SPARCS 发布页](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data) 给出 80 个 subset 和七类标签。Sentinel-2 的 10/20/60 m 波段来自 [Copernicus 官方文档](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html)，Landsat-8 波段分辨率来自 [USGS 官方说明](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites)。

### Deep Fmask 发布包实测

- 下载对象：PANGAEA 官方 `Nambiar-etal_2022_DeepFmask.zip`，2.1 MB 标签发布包；不含 Sentinel-2 SAFE 原始影像。
- SHA256：`eb53c2dd795c1541afcef4f8cf5398953748883c82a79d9512dfa3fbb28e7bcd`。
- `label_mapping.txt`：`0 Unlabelled / 1 Clear-Sky Land / 2 Cloud / 3 Shadow / 4 Snow / 5 Water`。
- SAFE 清单行数：validation `22`，test `23`。
- 上述是发布包核验结果，不是训练或模型结果。

## 3. 冻结层级概率定义

首选实现采用条件概率树，使一致性由结构保证：

```text
q(p | x)                         shared parent softmax
r_d(y | p, x)                    dataset-d conditional child softmax
P_d(y | x) = q(parent(y) | x) * r_d(y | parent(y), x)
sum_{y:parent(y)=p} P_d(y | x) = q(p | x)
```

此时不另加一个可被权重调参的“层级一致性损失”。若公平基线必须保留独立 native leaf head，则只允许使用预注册的 grouped-marginal KL/JSD：比较共享 `q(p|x)` 与 native leaf 概率按父类求和后的分布。两种实现不能在看到结果后切换。

所有 native label 必须恰好映射到一个父类或显式 ignore。`Fill/Unlabelled/nodata` 永不参与损失与指标。L8 `Shadows?=no` scene 不具备完整三父类监督资格；Phase 64 主实验不对其中单像元进行 clear/shadow 猜测。

## 4. 输入与公平比较

- Day 2 快速筛选主输入冻结为公共六波段：RGB+NIR+SWIR1+SWIR2。Sentinel-2 使用 B4/B3/B2/B8/B11/B12，Landsat-8 使用 B4/B3/B2/B5/B6/B7。
- Sentinel-2 统一到 20 m；Landsat-8 维持 30 m。模型看到相同光谱语义，不声称像元物理尺度相同。
- RGB 结果作为必报对照；Landsat 独有 pan/TIRS 与 Sentinel 独有 red-edge/cirrus 不进入 72 小时主门。
- 同一目标域内，五个方法使用相同 target-adapt 样本、像素有效掩膜、增强、优化器步数、checkpoint 规则和评价集。
- 每个目标域报告父类三类 IoU/mIoU、Shadow IoU、native 子类指标、source-val 父类与四类指标，以及训练参数量。
- Deep Fmask 的 snow/water、SPARCS 的各 surface 子类必须原样分层；总体三类分数不能替代这些条件结果。

## 5. Day 2 单 seed 最小基线

固定一个 seed 快速筛选以下五项，不做超参数搜索：

1. `source-only`：现有 CloudSEN 四类模型按固定求和映射到三父类，不读取目标训练标签。
2. `full fine-tuning`：全部参数可训练，作为容量上界；与参数高效方法使用相同 target-adapt 监督。
3. `LoRA`：必须是真正注入冻结骨干线性层的低秩增量。仓库现有 `LoRAReins` 是 Reins token 的低秩参数化，不能未经验证就冒充标准 LoRA。
4. `MsRE`：复用已验证的轻量残差载体；参数量和启用参数名必须机器统计。
5. `shared-parent-only`：只训练共享三父类 head，用于判断收益是否来自粗粒度共享本身；不报告不存在的 thin/thick 预测。

Day 1 代码盘点结论：MsRE 和 `LoRAReins` 已存在；真正标准 LoRA、可训练共享父类 head、Deep Fmask/SPARCS loader 尚未完成。因此在这些接口通过单元测试、数据 hash 和 scene 泄漏审计前，不启动 Day 2 正式比较。

## 6. Day 3 生死门

以下门同时满足才进入正式多 seed 与方法开发：

1. 至少两个独立目标域相对各自 source-only 的父类 mIoU 同方向提升；
2. 两个目标域父类 mIoU 增量的等权平均至少 `+2.0` 点；
3. CloudSEN source-val 父类 mIoU 下降不超过 `1.0` 点，同时原生四类结果必报；
4. MsRE 新增可训练参数维持既有约 `0.38M` 量级，硬上限暂冻结为 `0.50M`；
5. 每个目标域的 Shadow IoU 不得相对 source-only 下降超过 `2.0` 点，且不得出现接近零的灾难性坍缩；
6. 改善不能仅来自把 snow/water 等地表类并入 surface 后掩盖 native 子类失败。

任一失败则停止当前“数据组合 + 骨干 + 层级头”组合，不增加注意力模块、不调一致性权重。单 seed 只承担生死筛选，不能作为最终论文统计。

## 7. Day 1 决策

- **科学协议：通过。** 已找到一个源域、两个主目标候选和一个备份，且至少两个目标候选含原生 shadow；标签树可在不伪造 thin/thick 的前提下覆盖全部有效类。
- **两目标训练就绪：未通过。** 当前仓库只有 CloudSEN/L8 loader；Deep Fmask 只有标签发布包与清单，缺 SAFE 影像和 loader；SPARCS 同样未就绪。
- **下一动作：** 先实现 Deep Fmask manifest/loader、下载范围与 scene/tile overlap 审计，再实现共享父类条件树和标准 LoRA 基线。任何训练结果在第二目标域未就绪前都只能称 L8 单域调试，不能宣称通过 72 小时双目标门。
