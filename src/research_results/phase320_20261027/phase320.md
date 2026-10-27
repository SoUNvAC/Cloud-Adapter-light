# Phase320 实验记录

- 目标：一次交付错Shadow原生类别/边界/原几何状态分解，及PL单一全局阈值与CB默认点的取舍诊断。
- 改动：只读既有317/319缓存和验收参考；完整SPARCS60/L8fit100；原val CMC32、SPARCS10、L8963图/8scene；精确float32排序/完整并列块、原AP口径、独立真值与计数复核。
- 环境：gzs cloud-lite-pt210，CPU<=4线程；零训练、拟合、模型加载、前向、下载。
- 止损：CPU2h，另1h修复复验余量；16GiB、排序临时6GiB、启动磁盘>=8GiB、宿主空闲>=4GiB、新持久双机含预留<=1GiB；仅删除明确登记的本轮排序临时文件，无自动重试。
- 结果：详见RESULT/CONCLUSIONS与全部规定表。阈值为oracle_diagnostic，不冒充部署方法；标签不支持的暗地表/地形阴影/标注错误原因保持未知，历史窗口与候选结论不重写。
- 实际成本：{"status": "completed", "epoch": 1793087259.4293916, "seconds": 231.65129417798016, "stages": {"prepare": {"exit": 0, "seconds": 7.2381661320105195, "peak_tree_RSS": 0}, "test": {"exit": 0, "seconds": 2.003816102980636, "peak_tree_RSS": 3338240}, "errors": {"exit": 0, "seconds": 8.01301822194364, "peak_tree_RSS": 101015552}, "threshold": {"exit": 0, "seconds": 206.37215102498885, "peak_tree_RSS": 3810684928}, "audit": {"exit": 0, "seconds": 6.011304590967484, "peak_tree_RSS": 129826816}, "report": {"exit": 0, "seconds": 2.003784636966884, "peak_tree_RSS": 3497984}}, "resources": {"available_memory": 32697520128, "free_disk": 737401217024, "peak_RSS": 17207296}, "neural_calls": 0, "training_updates": 0, "scratch_remaining_files": []}
- 网络及完整备份：仅在本地流式回传与SHA实际验收后补真实回执；不设置定时任务，不自动执行321。

- 完整备份实际回执：47成员/9260212字节，1段及逐文件/整流SHA全部通过；payloadSHA256=8717df1c63c12945f696591542ee276ff4cd1a51e16838b94b20343495045c9d，manifestSHA256=f752e7fb02d81cad095ca6bbc6af5d599909cb1eec33ac7a61f58ba3b2192449。双机新持久含摘要预留118520424字节<=1GiB，无持久大归档。网络与失败实录如有则按原件保留；本机只编辑、文件SHA与备份。

- 实际主要结果：SPARCS错误30188，主要cloud18065/land8922；L8错误76886，主要clear69593/thin_cloud6498，clear原因仍未知。相同CB宏FPR预算下PL宏召回CMC86.5931%、SPARCS40.8403%、L848.5646%；CB原点84.0823%、44.7384%、50.3818%。CMC可复现，SPARCS/L8分别仍差3.8980/1.8173pp；仅oracle二分类诊断，不证明云特有创新。
- 网络实录：gzs到GitHub的Git GnuTLS和curl OpenSSL均报TLS失败，并有SSH握手拒绝；已推送同一提交72fcbbe通过38004字节Git bundle验SHA后由gzs实际git pull，原origin/凭据未改。传输辅助脚本曾在解析阶段因换行转义失败，未写远端或启动计算；原日志保留，修正后CPU首次执行成功，科学阶段无失败、无重复。
- 备份恢复：首次47文件及唯一压缩段全部SHA通过后，最终控制JSON解析失败；原始控制行未保存，精确成因不妄断。远端主回执completed且无活动传输，仅补取六个小元数据文件；整流SHA与已验唯一分段相同，原47成员再次按清单核验，未重传原始结果。首失败、恢复代码及证明保留delivery_evidence/。
- 报告仅澄清CB原argmax逐scene成本应查已锁定Phase317 PER_SCENE_METRICS；原始RESULT/复核JSON及SHA均保留，不改计算结果或既有报告。无定时任务，不自动321。
