# 实验记录

## 2026-10-15 · phasecal-1 / 既有 Phase122 接续

- 目标：执行 PHASEcal-1；单次 C100（seed12201、1000步），100训练组／53校准组；R与C100各冻结一个1%组均负类误报阈值，再固定评价 Catalogue32／PixBox16产品10组。
- 接手现场：已有控制器 PID183225、PixBox worker PID184175，训练及校准已封存，继续原作业，未重复训练。训练源码提交 b7ea7e8，接手代码 a016911；新增 report_phasecal1.py 仅从封存证据生成要求的一张交付表，不改变模型或阈值。
- 环境／网络：ssh gzs 可达；WSL2 PC-201807020355、RTX4090D。激活 cloud-lite-pt210 后沿用验收的 /home/scv/shared/phase111_ocm_runtime/runtime_cuda128/bin/python（Torch2.11.0+cu128）。默认本地Python缺NumPy，改用已有Codex内置Python，两项CPU测试本地及远端均通过，未安装包。沙箱内SSH曾无法解析别名，授权连接成功。
- 止损线：普通3500调用／7200秒，含恢复累计4375调用／9000秒；CPU16GiB、GPU20GiB、中间文件4e9 bytes、宿主I盘余量≥4e9 bytes；截止2026-10-20 00:00上海时间。异常保留现场，不重启控制器、不清理历史目录。
- 数据：本轮不涉及L8；后续若涉及L8，仅shadow?=1，排除shadow?=0。PixBox参考留本地，不上传。
- 当前证据：encoder参数／buffer不变，decoder更新，训练参数14,135,988；已冻结R阈值0.7693686485290528、C100阈值0.36361625790596014。评价尚未结束，此处不填效果。完成后补写实测结果与备份位置。
