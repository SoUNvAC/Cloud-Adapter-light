# Phase97：Source单调约束树，一次固定探索

## 1．数据结果

原100组拟合、53组留出校准、32组development原20m评价；仅新增一次树拟合、一个冻结阈值。
输入为B04/B03/B02/B08/B11/B12/Source logit；唯一模型参数变化是Source列单调递增约束。

|模型|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|超1%组数|最大组FPR %|
|---|---:|---:|---:|---:|---:|---:|---:|
| A_G7_100 | 73.7164 | 40.8613 | 38.8948 | 1.7762 | 4.5128 | 5 | 44.6105 |
| B_SourceMonotonic | 73.6984 | 47.4833 | 45.7913 | 1.9032 | 4.5841 | 6 | 44.2356 |
| SourceS_100 | 74.6569 | 20.8039 | 26.5473 | 1.0050 | 1.9888 | 5 | 14.1712 |

B−A ap：-0.018009个百分点，配对相关组95%描述区间[-2.798094457311296, 2.5234668474613895]，有效组数7。

B−A recall：6.896446个百分点，配对相关组95%描述区间[-0.2738656387397168, 15.94018610043989]，有效组数7。

B−A fpr：0.127017个百分点，配对相关组95%描述区间[-0.031075051156847732, 0.32964279048569556]，有效组数32。

四项固定筛选：
- AP_loss_at_most_1pp: True
- recall_loss_at_most_1pp: True
- macro_FPR_at_most_1percent: False
- maximum_group_FPR_at_most_10percent: False

T31TGF分数分位数及A-only/B-only/共同误报数量见T31TGF.json；全部组、Surface/Cloud分层见per_group.csv。

## 2．结论探讨

四项全部满足，可另立稳定性与预算迁移协议；任一失败，关闭本次固定单调约束方案，不追加调参。
本轮固定判定：close_this_fixed_monotonic_scheme
单调约束只规定其余六列固定时，Source升高不能使预测概率降低；低Source仍可能产生高预测值。无论本轮成败，都不能据此证明T31TGF原失效机制。
置信区间使用10000次相关组配对bootstrap、seed97012；工程筛选使用预先约定点估计，不是非劣证明或部署保证。
Exposed development; descriptive related-group intervals, not independent confirmation
Human B not performed;54 reused Phase89 blocks lack pre-crash SHA; original failed gates unchanged
Monotonicity does not require low probability at low Source score; no per-scene1percent guarantee
Reference labels not newly human confirmed; no three-class mIoU fabricated

成本：{"seconds": 429.3721750340046, "fit_seconds": 4.164225844986504, "calibration_scoring_and_threshold_seconds": 161.23932925000554, "development_loading_and_scoring_seconds": 196.86216442700243, "development_metrics_seconds": 52.900738885000465, "process_peak_rss_bytes": 3282530304, "cpu_threads_max": 4}
