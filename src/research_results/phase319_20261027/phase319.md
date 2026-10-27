# Phase319 实验记录

- 目标：太阳几何能否挑出更可靠且有足够保留量的Shadow。
- 改动：固定317缓存和160 fit图；真实MTL/原网格/原L8坐标；唯一200–12000m、±5度核、反方向及同k高分；先封存后独立CPU参考审计。零训练、拟合和前向。
- 环境：gzs cloud-lite-pt210、至多4CPU线程；已有rasterio投影/WGS84测地基准及往返QA/proc监控，未安装或升级依赖。
- 止损：CPU16GiB/3h，空闲RAM和磁盘>=4GiB，双机新持久含预留<=2GB；规则和投影近似准入误差均未降低。
- 首次失败：160产品坐标核验完成后，NumPy bool_不能写JSON；无几何选择/质量读取。原失败完整保留recovery01/；独立恢复仅转JSON标量，因原记录未写成而重复一次坐标身份检查，未改变参数或既有代码。
- 结果：not_testable；见GO_NO_GO/RESULT/完整160图缺项，not_testable不解释为几何无效。
- 实际资源：{"status": "completed", "epoch": 1793075079.3919818, "seconds_including_initial_failure": 94.4114410749171, "decision": "not_testable", "initial_stages": {"prepare": {"exit": 0, "seconds": 0.10317213996313512, "peak_tree_RSS": 0}, "inputs": {"exit": 1, "seconds": 12.019349011941813, "peak_tree_RSS": 98582528}}, "recovery_stages": {"inputs": {"exit": 0, "seconds": 8.012533080996946, "peak_tree_RSS": 98373632}, "geometry": {"exit": 0, "seconds": 52.13812009594403, "peak_tree_RSS": 240599040}, "quality": {"exit": 0, "seconds": 10.04065309499856, "peak_tree_RSS": 118255616}, "audit": {"exit": 0, "seconds": 10.04547046602238, "peak_tree_RSS": 62808064}, "report": {"exit": 0, "seconds": 2.0096323309699073, "peak_tree_RSS": 7286784}}, "resources": {"available_memory": 32694558720, "free_disk": 737794142208, "peak_RSS": 18001920}, "neural_calls": 0, "training_updates": 0, "recovery01_sha256": "a9c1228a0554a23b79ed2aefabbfa5bfcd3dc431919ee8042c15dba002927be4"}
- 网络与完整备份：仅在本地实际验收后补回执，原日志不覆盖。

- 本地完整备份实际验收：616成员，原始7923653字节，3段；所有成员/分段/整流SHA通过。整流SHA256=0973b9b466feba296aa480c1a2516f4a5ff3b97db58a97877cfc33f4644e7002，清单SHA256=0e9032189d3ba3e7f042712a14278ab84653fd427f9398566e06f989dd8534f7。双机含摘要预留115847306字节<=2GB，无持久大归档。本机仅编辑/文件SHA/备份。网络及执行探测实录见NETWORK_EVENTS.json；原stderr保留，未因此重跑科学阶段。

- 收尾澄清：SPARCS的not_testable源于本实现事前0.1度/0.1%局部线性投影准入覆盖不足，不能称太阳资料缺失。其子集指标不外推全池；L8独立no_go。原始RESULT全文及SHA保留在RESULT_SOURCE.md和DELIVERY_REVIEW.json，未改远端原始结果或任何选择规则。
