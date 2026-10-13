# Phase100：联合误报约束的校准迁移

## 1．数据结果

仅原53组校准三个T_tail，落锁后原32组development一次评价六工作点。零拟合／网络前向／下载；非独立确认。

|评分|阈值|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|最大FPR %|超1%组|零召回组|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|G7_100|old|73.7164|40.8613|38.8948|1.7762|4.5258|44.6105|5|1|
|G7_100|T_tail|73.7164|40.8613|32.8041|1.0846|2.8705|29.0161|3|1|
|SourceS_100|old|74.6569|20.8039|26.5473|1.0050|1.9754|14.1712|5|2|
|SourceS_100|T_tail|74.6569|20.8039|11.5673|0.4136|0.8954|7.2731|2|5|
|MIN|old|75.5384|34.2932|40.9585|0.8835|1.9865|18.3898|5|0|
|MIN|T_tail|75.5384|34.2932|31.8023|0.5537|1.2931|12.2152|3|1|

MIN_T_tail_minus_SourceS_T_tail：宏召回差20.234939个百分点，配对相关组95%描述区间[7.551813629584502, 35.408779244970816]，7组。

G7_T_tail_minus_SourceS_T_tail：宏召回差21.236766个百分点，配对相关组95%描述区间[5.473519836252594, 42.88754770731159]，7组。

G7_100_tail_minus_old：宏召回差-6.090741个百分点，配对相关组95%描述区间[-8.892094187045444, -3.1173853345692577]，7组。

SourceS_100_tail_minus_old：宏召回差-14.979912个百分点，配对相关组95%描述区间[-28.12098088108481, -3.753035441033305]，7组。

MIN_tail_minus_old：宏召回差-9.156214个百分点，配对相关组95%描述区间[-14.892282006405727, -4.232062351982981]，7组。

bootstrap10000次，seed100013；召回/AP仅有参考云影组，FPR全部有负类组；无参考云影组为NA。
同评分新旧AP完全不变。实际误报率不同；不能称完全相同实际误报下比较。

## 2．结论探讨

end_global_threshold_repair_route
筛选：{"G7_100": {"criteria": {"macro_fpr_at_most_1percent": false, "max_group_fpr_at_most_10percent": false, "recall_gain_at_least_5pp": true, "macro_AP_loss_at_most_1pp": true}, "all_four_pass": false, "delta_recall_pp": 21.236766123374817, "delta_AP_pp": -0.9405502387632159, "decision": "close_fixed_tail_calibration_candidate"}, "MIN": {"criteria": {"macro_fpr_at_most_1percent": true, "max_group_fpr_at_most_10percent": false, "recall_gain_at_least_5pp": true, "macro_AP_loss_at_most_1pp": true}, "all_four_pass": false, "delta_recall_pp": 20.234939048241422, "delta_AP_pp": 0.881463141956329, "decision": "close_fixed_tail_calibration_candidate"}}
SourceS预算：{"macro_point_at_most_1percent": true, "max_group_at_most_10percent": true, "macro_upper95_at_most_1percent": true}
点估计达标不代表稳定满足预算；单侧FPR上界完整报告。Phase98失败结论不变，未选择部署赢家。
Phase99 development oracle只在全部六行评价完成后另表对照，不能将差额直接称可恢复校准收益。
Repeatedly exposed development;53 calibration maximum is only empirical
Reference semantics not newly human confirmed; human B not performed;54 reused Phase89 blocks lack pre-crash SHA
Actual false-positive rates differ across strategies; not equal actual FPR comparison
Oracle gap is not achievable deployment calibration gain; no deployment winner selected

成本：{"seconds": 477.78453858502326, "calibration_seconds": 304.71972667000955, "development_seconds": 172.837707114988, "metrics_seconds": 59.87920262498665, "peak_rss_bytes": 5980782592, "cpu_threads_max": 4}
