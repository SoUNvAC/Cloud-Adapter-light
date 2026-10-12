# Phase98：冻结分数AVG／MIN组合

## 1．数据结果

原53组负类校准两个阈值，原32组development评价，零拟合／零网络前向／零下载。组合与组合AP全程float64。

|策略|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|超1%组数|最大FPR %|
|---|---:|---:|---:|---:|---:|---:|---:|
| G7_100 | 73.7164 | 40.8613 | 38.8948 | 1.7762 | 4.5193 | 5 | 44.6105 |
| SourceS_100 | 74.6569 | 20.8039 | 26.5473 | 1.0050 | 1.9760 | 5 | 14.1712 |
| AVG | 76.1066 | 38.2996 | 42.2427 | 1.4149 | 3.1759 | 5 | 29.0997 |
| MIN | 75.5384 | 34.2932 | 40.9585 | 0.8835 | 1.9625 | 5 | 18.3898 |

AVG_minus_SourceS_100 ap: 1.449653个百分点，95%描述区间[-2.9853403006271235, 5.053091766243877]，7组。

AVG_minus_SourceS_100 recall: 15.695492个百分点，95%描述区间[6.693647927428517, 25.62944263771848]，7组。

AVG_minus_SourceS_100 fpr: 0.409902个百分点，95%描述区间[-1.2099279752321281, 2.7171181367196517]，32组。

MIN_minus_SourceS_100 ap: 0.881463个百分点，95%描述区间[0.26548345185745675, 1.6154973222002782]，7组。

MIN_minus_SourceS_100 recall: 14.411241个百分点，95%描述区间[6.7625156378455396, 22.078648432204822]，7组。

MIN_minus_SourceS_100 fpr: -0.121531个百分点，95%描述区间[-0.9427863974032566, 0.47175826205320726]，32组。

MIN_minus_AVG ap: -0.568190个百分点，95%描述区间[-3.908350424537545, 3.5874724893926637]，7组。

MIN_minus_AVG recall: -1.284251个百分点，95%描述区间[-9.249319845950565, 9.385041139422526]，7组。

MIN_minus_AVG fpr: -0.531433个百分点，95%描述区间[-2.7654618082323963, 1.1268685577228217]，32组。

固定筛选结果：{"AVG": {"criteria": {"AP_loss_at_most_1pp": true, "recall_gain_at_least_5pp": true, "macro_FPR_at_most_1percent": false, "maximum_group_FPR_at_most_10percent": false}, "all_four_pass": false}, "MIN": {"criteria": {"AP_loss_at_most_1pp": true, "recall_gain_at_least_5pp": true, "macro_FPR_at_most_1percent": true, "maximum_group_FPR_at_most_10percent": false}, "all_four_pass": false}}

## 2．结论探讨

判定：close_both_fixed_fusion_branches；通过分枝：[]
每分枝四项全部满足才可另立协议；本轮不追加权重、公式、阈值或拟合。池化AP改善不替代宏AP／召回／误报条件。
组合改变排序，已重新计算AP；组合评分不称已校准后验概率。MIN逐点小于两输入，不意味着使用自身校准阈值后误报必然更少。
T31TGF单独分位数与误报见T31TGF.json；全部32组和Surface／Cloud分层保留。无参考云影组AP／召回为NA，继续参与误报评价。
Repeatedly exposed development; descriptive related-group intervals only
Reference labels not newly human confirmed
Human B not performed;54 reused Phase89 probability blocks lack pre-crash SHA; original failed gates unchanged
Independent fusion thresholds may be lower; MIN score bound does not guarantee fewer false positives than SourceS

成本：{"seconds": 375.9188463370083, "calibration_scoring_and_threshold_seconds": 182.14632621100463, "development_loading_and_scoring_seconds": 112.05064546001086, "metrics_seconds": 81.64181585599727, "peak_rss_bytes": 5146480640, "cpu_threads_max": 4}
