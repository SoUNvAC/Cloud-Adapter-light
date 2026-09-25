# Phase 61D2 校准盲法复核操作手册

版本：1.0。两位 reviewer 必须使用同一版本；主复核开始前须锁定本手册哈希与校准结论。

## 1. 盲态与可见信息

Reviewer 不得查看原始标签、模型预测、模型名称、实验方法、错误来源层或 sealed manifest。Reviewer 必须查看每个单元的完整证据页：Landsat 8 B1–B11、4-3-2 真彩色、5-4-3 彩红外、7-6-4 SWIR 合成、9-1-2 cirrus/coastal/blue 合成、太阳方位/高度、采集时间、位置与 biome。黄色角标只定位待判像元。

证据页采用每波段在完整 512×512 图块上的 2–98% 显示拉伸。它只帮助目视解释，不能把显示亮度当绝对反射率或温度。B8 原生 15 m；B10/B11 原生 100 m 且产品可重采样至 30 m，不能把热红外的细边界当作真实 30 m 边界。USGS 波段与分辨率定义见 [Landsat 波段表](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites)。太阳方位角为场景中心相对真北顺时针角，定义见 [USGS Collection 2 数据字典](https://www.usgs.gov/centers/eros/science/landsat-collection-2-data-dictionary)。

## 2. 标签词表

- `clear`：待判像元无可辨识云、薄云/雾霾或阴影污染。
- `thin_cloud`：半透明云，地表纹理仍可见；优先结合 cirrus、NIR、SWIR 证据。
- `thick_cloud`：高反射、明显不透明或地表被遮蔽的云体。
- `cloud_shadow`：暗像元且存在与云体、太阳反方向和合理位移一致的云—影关系。
- `terrain_water_shadow`：暗像元更可由地形遮蔽、水体、岸线或局部照明解释，缺少可信云—影关联。
- `haze_cirrus`：弥散雾霾或高层卷云证据存在，但不足以按操作阈值定为 thin cloud。
- `boundary_mixed`：目标正落在一像素级边界、配准差或混合像元上，无法给出稳定的内部类别。
- `unobservable_nodata`：所有可用证据均无有效信息（例如纯色、填充值或严重饱和）。
- `uncertain`：证据有效，但无法把不确定性约束为两个具体语义类别。不得把它当作省事选项。

允许用竖线提交两个类别的最小集合，例如 `thin_cloud|haze_cirrus`、`clear|cloud_shadow`、`cloud_shadow|terrain_water_shadow` 或 `thin_cloud|thick_cloud`。集合最多两个语义标签；`boundary_mixed`、`unobservable_nodata`、`uncertain` 必须单独使用。禁止空值；无法评估时必须写 `unobservable_nodata`。

## 3. 固定判读优先级

1. 先判断证据是否有效；无信息则 `unobservable_nodata`。
2. 再判断目标是否为一像素边界/配准/混合问题；若是则 `boundary_mixed`，不要强迫成内部类别。
3. 判断 cloud/contaminated 与 non-cloud。
4. 对云体再区分 thin 与 thick；若仅阈值不确定，用 `thin_cloud|thick_cloud`。
5. 对暗区先检查太阳反方向及附近云体是否形成合理云—影关联；只有几何证据支持才判 `cloud_shadow`。
6. 若地形、水体、岸线或照明解释更强，判 `terrain_water_shadow`；二者均合理时用集合标签。
7. 区分 haze/cirrus 与 thin cloud；二者仍等可能时用集合标签。
8. 能约束为两个类别时必须用最小集合；只有不能约束时才用 `uncertain`。

太阳几何只提供方向约束，不提供云高、DEM 或时序，因此不能单独证明 cloud shadow。证据页蓝线显示与太阳方位相反的预期投影方向；若 north-up 校验失败，生成程序会停止而不是画误导箭头。

## 4. 校准与独立复核

1. 两位 reviewer 先分别填写 40 个校准单元。
2. 共同讨论所有分歧、典型反例及上述优先级，填写 `calibration_consensus.csv`；这 40 个单元永久排除于最终一致性统计。
3. 用 `lock_phase61d2_calibration.py` 锁定两份原始校准判断、共识、手册哈希和讨论纪要。若修改本手册，必须重新生成并锁定复核包。
4. 锁定后，两位 reviewer 独立填写主复核 CSV，不得讨论，不得查看对方判断；保留两份原始文件。
5. 评分程序只把实际 A/B 分歧交给第三位专家。第三位专家可见两份判断与同一证据页，但仍对原标签、模型和方法盲态，并须写裁决理由。

主复核含 160 个旧错误分层单元与 50 个此前未见图块的确认单元；二者会混合随机编号，reviewer 不知道单元来源。确认集不按模型错误或置信度筛选。

## 5. 预注册统计与硬门

最终统计仅使用 210 个主复核单元，同时分别报告 160 个独立集、50 个确认集和各 biome：

- 整个集合标签作为类别的 exact agreement、Cohen κ、Gwet AC1；
- 集合有交集的 compatible agreement；
- Thin 与 cloud-shadow 的 reviewer membership IoU；
- 分歧类型：边界、thin↔thick、clear↔cloud-shadow、cloud-shadow↔terrain/water-shadow、thin↔shadow；
- 第三方裁决后，原标签与最终标签的四类 membership IoU。

硬门按最保守口径自动执行：κ 或 AC1 任一低于 0.40，四类 mIoU 不得作为主指标；Thin 或 Shadow reviewer IoU 任一低于 0.30，终止四类硬标签主线；两类 reviewer IoU 均超过 0.60、但原标签—裁决标签任一低于 0.40，支持标注过程偏移方向；两类 reviewer 与原标签 IoU 均至少 0.60，则回到模型问题。其余情况报告为混合证据，不强行归因。
