# Phase77：B04重采样链有限溯源

## 1 数据结果

执行完成，EXIT_CODE=0，0新增原始影像下载。复用Phase75/76固定三景、6个既有已校验原JP2及3个fit cube；不读取标签、模型分数或确认像元。

三景Phase76的B11/B04整图指标、固定窗口指标和完整残差均精确复现；全部复现通过后才开始新候选。B11内裁65/原方向不变。首次新候选残差前保存candidate_lock.json，固定12个候选、10个数学等价类、25选择/24留出点、17×17窗口及原误差/支持门。

三景各0/12个候选通过25点选择门；因此没有候选获得独立留出核验资格。为完整交付，全部12个候选的留出及整图误差仍报告为诊断，不允许据此重新晋级。所有候选所有固定点及全图的双方有效率/有效性一致率均100%，每分组25/24个可用点；失败不是支持不足。

三个范围的half-pixel逐值完全等价（每景两项比较，6/6检查通过，最大差值0），并精确复现Phase76基线。高斯、宽三角、align-corners和四个整数相位均没有通过。

下表仅展示整图RMSE最低的固定诊断对照integer_0_0（原tile行列均从0起，每2像元抽取），不是选出的成功处理链。误差为TOA单位：

| 场景 | 中位绝对误差 | P99绝对误差 | RMSE | 状态 |
|---|---:|---:|---:|---|
| T37LDL | 0.001100 | 0.009600 | 0.002823 | unresolved |
| T20PMQ | 0.000700 | 0.005000 | 0.001512 | unresolved |
| T46RBU | 0.001200 | 0.021700 | 0.004961 | unresolved |
| 原误差上限 | 0.000050 | 0.000200 | 未设单独门 | — |

即使最接近的整数对照，中位误差仍为门槛的约14–24倍，P99约25–108.5倍。全部12候选原始数值见candidate_summary.csv；逐点和四象限完整保存。

## 2 结论分析

本轮预先指定的采样中心、抗混叠/宽三角和整数相位候选，不能在工程容差内解释B04与发布cube的差异。integer_0_0误差较小只是有限诊断结果，既不能证明最近邻是作者流程，也不能改写README的双线性描述。

三景均为unresolved，按协议关闭本轮重采样搜索，不增加候选、拟合gain/offset、放宽界限或改B11位置。该结果不排除其他尚未测试的发布预处理流程；具体来源仍未知。

保留边界：**B11坐标证据可用，RGB预处理溯源未闭合。** 不宣布B04通过、全六波段映射通过或全185景通过，不进行方向信息/云影投影实验。若以后使用B11开展限定方向实验，需要另立协议。本轮无训练、预测或方法性能结论。

## 文件与复核

- candidate_lock.json：候选/公式/等价类/NoData/支持门、输入和代码SHA、软件版本；高斯使用sigma0.5/truncate4的代数等价6tap融合实现，只在原tile真实支持上计算，无反射补边。
- candidate_summary.csv / scene_status.csv：全部候选整图数值与逐景状态；validity_audit.csv：双方有效性/不一致计数。
- partition_metrics.csv：从已保存原始残差汇总25/24点分组误差，无新候选计算或重新选择。逐点和整图相关系数保留于原始CSV。
- 每景目录：baseline_reproduction.json、selection_points/decisions.csv、selection_freeze.json、all_points.csv、full_metrics.csv、spatial_metrics.csv、mathematical_equivalence.json、status.json，以及12个完整残差NPY。
- artifact_hashes.json：远端报告SHA；LOCAL_VERIFIED.json：本地逐文件、Git blob、Phase76输入及36个残差全图统计独立复核。
- PID/GIT_COMMIT/EXIT_CODE/console.log/error.log：shared控制日志；本地和远端7项合成检查通过。

仅取回shared审计产物，JP2留远端，无work_dirs同步。远端结果：/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/phase77_20261011/。历史Phase76失败报告保持不变，预算截止2026-10-20北京时间。
