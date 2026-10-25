# Data profile: /home/scv/shared/phase311_20261025/TRAINING_CURVE.csv

**Shape:** 1000 rows × 27 cols

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
| `lambda_CM` | boolean | 1000 | 0 | 1 levels: 1.0(1000); min_group_n=1000 |
| `PL` | continuous | 955 | 45 (4%) | mean=0.385, sd=0.214, range=[0.0129, 1.68], skew=0.45 (approximately symmetric); outliers=9 (IQR); -> log axis |
| `CM` | continuous | 955 | 45 (4%) | mean=0.0414, sd=0.454, range=[0, 11.8], skew=20.68 (highly skewed); outliers=198 (IQR) |
| `weighted_CM` | continuous | 955 | 45 (4%) | mean=0.0414, sd=0.454, range=[0, 11.8], skew=20.68 (highly skewed); outliers=198 (IQR) |
| `total` | continuous | 955 | 45 (4%) | mean=0.427, sd=0.486, range=[0.0131, 11.9], skew=15.59 (highly skewed); outliers=18 (IQR); -> log axis |
| `lora_grad_norm` | continuous | 955 | 45 (4%) | mean=4.11, sd=13, range=[0.000161, 196], skew=8.95 (highly skewed); outliers=97 (IQR); -> log axis |
| `PL_grad_norm` | continuous | 4 | 996 (100%) | mean=1.14, sd=1.36, range=[0.0325, 2.91], skew=0.31 (approximately symmetric) |
| `CM_grad_norm` | continuous | 4 | 996 (100%) | mean=0.333, sd=0.381, range=[0, 0.88], skew=0.55 (moderately skewed); outliers=1 (IQR) |
| `active_count` | continuous | 1000 | 0 | mean=4.74e+03, sd=1.98e+04, range=[0, 2.21e+05], skew=6.58 (highly skewed); outliers=201 (IQR) |
| `active_fraction` | continuous | 482 | 518 (52%) | mean=0.247, sd=0.314, range=[0, 1], skew=1.23 (highly skewed); outliers=30 (IQR) |
| `mT_mean` | continuous | 482 | 518 (52%) | mean=-0.511, sd=0.63, range=[-3.29, 1.45], skew=0.39 (approximately symmetric); outliers=18 (IQR) |
| `mS_mean` | continuous | 482 | 518 (52%) | mean=-0.858, sd=0.922, range=[-4.95, 3.38], skew=-0.17 (approximately symmetric); outliers=99 (IQR) |
| `d_mean` | continuous | 482 | 518 (52%) | mean=0.0692, sd=0.254, range=[0, 3.41], skew=7.83 (highly skewed); outliers=74 (IQR) |
| `d_max` | continuous | 482 | 518 (52%) | mean=0.366, sd=0.526, range=[0, 4.22], skew=2.61 (highly skewed); outliers=36 (IQR) |

## Warnings
- 列 'PL_grad_norm' 缺失率 100% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'CM_grad_norm' 缺失率 100% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'active_fraction' 缺失率 52% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'mT_mean' 缺失率 52% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'mS_mean' 缺失率 52% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'd_mean' 缺失率 52% — 画图前考虑是否填补、剔除、或在图注中交代。
- 列 'd_max' 缺失率 52% — 画图前考虑是否填补、剔除、或在图注中交代。

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['attempt', 'A', 'P', 'Surface', 'Cloud']）或 pairplot 散点矩阵
- attempt 跨数个量级（1 ~ 1e+03）→ 用对数 y 轴
- P 高度偏态（skew=2.86）→ 考虑对数变换或小提琴图代替均值柱图
- Shadow 高度偏态（skew=3.30）→ 考虑对数变换或小提琴图代替均值柱图
- PL 跨数个量级（0.0129 ~ 1.68）→ 用对数 y 轴
- CM 高度偏态（skew=20.68）→ 考虑对数变换或小提琴图代替均值柱图
- weighted_CM 高度偏态（skew=20.68）→ 考虑对数变换或小提琴图代替均值柱图
- total 跨数个量级（0.0131 ~ 11.9）→ 用对数 y 轴
- lora_grad_norm 跨数个量级（0.000161 ~ 196）→ 用对数 y 轴
- active_count 高度偏态（skew=6.58）→ 考虑对数变换或小提琴图代替均值柱图
- active_fraction 高度偏态（skew=1.23）→ 考虑对数变换或小提琴图代替均值柱图
- d_mean 高度偏态（skew=7.83）→ 考虑对数变换或小提琴图代替均值柱图
- d_max 高度偏态（skew=2.61）→ 考虑对数变换或小提琴图代替均值柱图

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。