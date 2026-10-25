# Data profile: /home/scv/shared/phase310_20261025/TRAINING_CURVE.csv

**Shape:** 1000 rows × 21 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `attempt` | continuous | 1000 | 0 | mean=500, sd=289, range=[1, 1e+03], skew=0.00 (approximately symmetric); -> log axis |
| `product` | text | 1000 | 0 |  |
| `A` | continuous | 1000 | 0 | mean=1.81e+05, sd=8.5e+04, range=[0, 2.62e+05], skew=-0.96 (moderately skewed) |
| `P` | continuous | 1000 | 0 | mean=2.21e+04, sd=4.92e+04, range=[0, 2.55e+05], skew=2.86 (highly skewed); outliers=135 (IQR) |
| `Surface` | continuous | 1000 | 0 | mean=9.67e+04, sd=1.16e+05, range=[0, 2.62e+05], skew=0.55 (moderately skewed) |
| `Cloud` | continuous | 1000 | 0 | mean=8.16e+04, sd=9.05e+04, range=[0, 2.54e+05], skew=0.55 (moderately skewed) |
| `Shadow` | continuous | 1000 | 0 | mean=3.04e+03, sd=7.06e+03, range=[0, 4.45e+04], skew=3.30 (highly skewed); outliers=176 (IQR) |
| `A_sha256` | text | 1000 | 0 |  |
| `P_sha256` | text | 1000 | 0 |  |
| `updated` | boolean | 1000 | 0 | 2 levels: True(955), False(45); min_group_n=45 |
| `teacher_calls` | boolean | 1000 | 0 | 2 levels: 1(955), 0(45); min_group_n=45 |
| `student_calls` | boolean | 1000 | 0 | 2 levels: 1(955), 0(45); min_group_n=45 |
| `lr` | continuous | 1000 | 0 | mean=0.0001, sd=2.71e-20, range=[0.0001, 0.0001], skew=-1.00 (moderately skewed) |
| `lambda_SP` | boolean | 1000 | 0 | 1 levels: 1.0(1000); min_group_n=1000 |
| `PL` | continuous | 955 | 45 (4%) | mean=0.302, sd=0.229, range=[0.00905, 1.33], skew=0.39 (approximately symmetric); outliers=3 (IQR); -> log axis |
| `SP` | continuous | 955 | 45 (4%) | mean=0.0427, sd=0.117, range=[0, 1.59], skew=6.41 (highly skewed); outliers=152 (IQR) |
| `weighted_SP` | continuous | 955 | 45 (4%) | mean=0.0427, sd=0.117, range=[0, 1.59], skew=6.41 (highly skewed); outliers=152 (IQR) |
| `total` | continuous | 955 | 45 (4%) | mean=0.345, sd=0.228, range=[0.00905, 1.62], skew=0.51 (moderately skewed); outliers=5 (IQR); -> log axis |
| `lora_grad_norm` | continuous | 955 | 45 (4%) | mean=2.8, sd=4.64, range=[0.000174, 55.3], skew=4.27 (highly skewed); outliers=83 (IQR); -> log axis |
| `PL_grad_norm` | continuous | 4 | 996 (100%) | mean=1.03, sd=1.02, range=[0.0325, 2.03], skew=0.00 (approximately symmetric) |
| `SP_grad_norm` | continuous | 4 | 996 (100%) | mean=0.459, sd=0.617, range=[1.2e-09, 1.33], skew=0.52 (moderately skewed) |

## Warnings
- 列 'PL_grad_norm' 缺失率 100% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'SP_grad_norm' 缺失率 100% — 画图前考虑是否填补、剔除、或在图注中交代。

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['attempt', 'A', 'P', 'Surface', 'Cloud']）或 pairplot 散点矩阵
- attempt 跨数个量级（1 ~ 1e+03）→ 用对数 y 轴
- P 高度偏态（skew=2.86）→ 考虑对数变换或小提琴图代替均值柱图
- Shadow 高度偏态（skew=3.30）→ 考虑对数变换或小提琴图代替均值柱图
- PL 跨数个量级（0.00905 ~ 1.33）→ 用对数 y 轴
- SP 高度偏态（skew=6.41）→ 考虑对数变换或小提琴图代替均值柱图
- weighted_SP 高度偏态（skew=6.41）→ 考虑对数变换或小提琴图代替均值柱图
- total 跨数个量级（0.00905 ~ 1.62）→ 用对数 y 轴
- lora_grad_norm 跨数个量级（0.000174 ~ 55.3）→ 用对数 y 轴

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。