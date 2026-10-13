# Phase99：平均与尾部误报约束下的召回可行性

## 1．数据结果

原53组校准池与32组development分别扫描；12个oracle点及6个原冻结点。零拟合／网络前向／下载。所有oracle不可部署。

|池|评分|工作点|宏召回 %|宏FPR %|最大FPR %|超1%组数|零召回组数|
|---|---|---|---:|---:|---:|---:|---:|
|calibration|G7_100|frozen|40.8893|0.9993|34.4687|7|2|
|calibration|SourceS_100|frozen|27.8410|1.0000|20.2069|10|1|
|calibration|MIN|frozen|39.1077|0.9906|19.1169|8|1|
|calibration|G7_100|O_mean|40.8893|0.9993|34.4687|7|2|
|calibration|G7_100|O_tail|34.6099|0.3052|5.4236|5|2|
|calibration|SourceS_100|O_mean|27.8411|1.0000|20.2069|10|1|
|calibration|SourceS_100|O_tail|11.3624|0.2983|10.0000|3|6|
|calibration|MIN|O_mean|39.1080|0.9906|19.1169|8|1|
|calibration|MIN|O_tail|31.9107|0.3772|9.9999|4|1|
|development|G7_100|frozen|38.8948|1.7762|44.6105|5|1|
|development|SourceS_100|frozen|26.5473|1.0050|14.1712|5|2|
|development|MIN|frozen|40.9585|0.8835|18.3898|5|0|
|development|G7_100|O_mean|31.8564|0.9997|27.1596|2|1|
|development|G7_100|O_tail|30.2951|0.1554|2.0031|2|1|
|development|SourceS_100|O_mean|26.4072|1.0000|14.1185|5|2|
|development|SourceS_100|O_tail|17.1882|0.6361|10.0000|4|3|
|development|MIN|O_mean|44.3296|1.0000|20.4087|5|0|
|development|MIN|O_tail|27.3845|0.4490|10.0000|3|2|

主要比较 development O_tail MIN−SourceS：10.196353个百分点，7个有参考云影组。

所有阈值下原分数宏／池化AP完全不变；全表含精确率、Shadow IoU、分层误报、最紧场景和逐组差。无新显著性检验。

## 2．结论探讨

存在值得另立校准研究的经验工作点。
Phase98两分枝继续关闭。Oracle阈值使用本池标签，不能用于部署、跨池或声称预算迁移成立。
Repeatedly exposed development and calibration are not independent confirmation
Oracle uses same-pool reference labels; gain is not achievable deployment calibration gain
Reference semantics not newly human confirmed; human B not performed;54 recovered Phase89 blocks lack pre-crash SHA
Original failed gates remain unchanged

成本：{"seconds": 488.71034658800636, "peak_rss_bytes": 5969944576, "cpu_threads_max": 4}
