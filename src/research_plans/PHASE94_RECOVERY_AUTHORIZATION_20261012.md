# Phase94故障恢复授权记录（2026-10-12）

原协议：PHASE94_HELDOUT_GROUP_CALIBRATION_20261012.md。原协议文件保持原SHA，不回写覆盖。

首轮100组输入核验完成，G7_100与SourceS_100均完成一次拟合；SourceS在保存前因验收代码要求重复归一化后的标准化均值/尺度逐位相等而中止，EXIT_CODE=1，首轮PID已退出。此时尚未读取53校准组标签或外部评价像元。G7已保存，其SHA256为：

`2e11a30996b04768b03d6668346d8a34ffda4100c93bbeded7d684e59091c6e6`

用户通过本线程明确答复：**“允许，仅补算SourceS一次并完成Phase94”**。

据此仅新增一次SourceS拟合，实际共3次拟合尝试、最终2模型。G7不再拟合；原100/53划分、样本索引/标签/输入哈希、原模型配置/优化器/约束/正则、四阈值exact-ties规则、评价池、统计规则均不变。不增加第三个模型或第三种校准。

恢复程序先核对原分组和样本哈希，再对SourceS补算并先保存系数；按优化器相同的归一化顺序复算标准化验证。原首轮代码SHA与修复后的SHA分别写入repair_authorization_lock.json，不覆盖原split_input_lock.json或fit_input_lock.json。原控制目录console.log/error.log/EXIT_CODE保留，新控制文件用resume_console.log/resume_error.log/RESUME_EXIT_CODE。

全部四阈值锁定后才评价development和ALCD。首轮未保存的模型成本不捏造；分别报告恢复进程与评价进程已实测成本。原失败门、人工B未执行和恢复概率缓存的谱系局限不变。
