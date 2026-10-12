# Phase94：训练内与留出组校准

## 1．数据结果

100组拟合、53组校准；两个全新模型，四个冻结阈值。零网络训练、零网络前向、零下载。

|评价池|策略|宏AP %|宏召回 %|宏精确率 %|宏FPR %|FPR单侧95%上界 %|超预算组|Shadow IoU %|池化AP %|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development | G7_100_T_in | 73.7164 | 80.6490 | 18.7243 | 5.9558 | 11.7638 | 8 | 16.3507 | 40.8613 |
| development | G7_100_T_cal | 73.7164 | 38.8948 | 34.7493 | 1.7762 | 4.5060 | 5 | 16.2802 | 40.8613 |
| development | SourceS_100_T_in | 74.6569 | 31.8772 | 39.1274 | 1.1898 | 2.3044 | 5 | 14.0051 | 20.8039 |
| development | SourceS_100_T_cal | 74.6569 | 26.5473 | 38.3225 | 1.0050 | 1.9738 | 5 | 12.7077 | 20.8039 |
| original_main | G7_100_T_in | 64.7530 | 88.0769 | 37.7422 | 4.9220 | 6.9059 | 9 | 35.5590 | 59.4131 |
| original_main | G7_100_T_cal | 64.7530 | 37.2599 | 74.8660 | 0.8941 | 1.5521 | 3 | 30.9193 | 59.4131 |
| original_main | SourceS_100_T_in | 63.9659 | 41.3334 | 80.2571 | 0.5702 | 1.1289 | 2 | 36.0332 | 51.6643 |
| original_main | SourceS_100_T_cal | 63.9659 | 33.4231 | 82.2126 | 0.4412 | 0.8968 | 2 | 29.8772 | 51.6643 |
| original_common | G7_100_T_in | 71.1849 | 88.3574 | 38.7846 | 3.7139 | 5.9148 | 5 | 36.4544 | 59.6073 |
| original_common | G7_100_T_cal | 71.1849 | 38.9630 | 77.4299 | 0.5960 | 1.4053 | 1 | 33.5178 | 59.6073 |
| original_common | SourceS_100_T_in | 68.9963 | 43.8205 | 86.8551 | 0.2713 | 0.6871 | 1 | 40.7975 | 54.0644 |
| original_common | SourceS_100_T_cal | 68.9963 | 34.4074 | 89.8237 | 0.1955 | 0.5293 | 1 | 33.0729 | 54.0644 |
| revised_common | G7_100_T_in | 69.5296 | 82.1672 | 47.1700 | 3.6165 | 5.9031 | 5 | 41.9928 | 57.2724 |
| revised_common | G7_100_T_cal | 69.5296 | 35.5387 | 76.9809 | 0.5979 | 1.4050 | 1 | 30.5836 | 57.2724 |
| revised_common | SourceS_100_T_in | 65.7722 | 34.2045 | 87.4631 | 0.2815 | 0.7016 | 1 | 32.1344 | 51.1465 |
| revised_common | SourceS_100_T_cal | 65.7722 | 27.2913 | 89.4830 | 0.2037 | 0.5428 | 1 | 26.1553 | 51.1465 |

## 2．结论探讨

同模型T_cal−T_in比较只涉及工作点，不是排序恢复；两阈值间逐组与池化AP完全一致。
留出校准内≤1%不保证未见组或跨数据集≤1%。误报降低须同时查看召回代价。两模型FPR不同不能仅按召回选择赢家。
原153组模型仅历史参考，不能用新旧差异隔离校准作用。拟合组数、校准组差异、跨域及60m聚合均可能影响。
所有区间为相关组bootstrap10000次、seed94012的描述性区间；已暴露评价池不是独立确认。
Human B not performed; original empty fields unchanged
54 Phase89 reused probability blocks lack pre-crash SHA
Exposed development and ALCD pools; no independent confirmation
Threshold change cannot improve ranking; no synthetic three-class mIoU
100-group models never refit on 153 groups

实际拟合尝试3次，最终保留2模型；若有修复详见repair_authorization_lock.json，未丢失或重拟合G7。
完整同模型配对区间、最差召回损伤和最大FPR增加见phase94_report.json；全部逐组值见per_group.csv和paired_changes.csv。
