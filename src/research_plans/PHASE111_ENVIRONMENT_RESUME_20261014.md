# Phase111环境恢复补充：用户解除环境4GB上限

用户2026-10-14明确授权“允许突破上限，继续执行”。解除原协议的新环境含安装缓存4GB上限；原预算失败记录及执行前锁保留。宿主物理卷至少4GB余量、单景输入、模型代码/HF/V1双权重、零标签/零训练/零拟合/零性能评价、4GB中间文件、4GB模型及波段下载、16GiB RAM、20GiB VRAM、1 GPU小时、模型主/重复调用累计256、两景块各模型重复一次等限制不变。其余15景不自动启动。

此前PyPI自动解析的Torch2.14.1依赖CUDA13，远端现有驱动576.88低于CUDA13要求的580系列。不得修改宿主驱动或原conda环境。安装/模型构建前重锁官方CUDA12.8构建Torch2.11.0+cu128和torchvision0.26.0+cu128，仍满足作者Torch>=2.3；这用于适配现有硬件，不通过更换模型权重/作者代码或看预测选择版本。未安装、未前向时完成环境修订，不改旧阶段结果或严格论文复现边界。

来源：[NVIDIA兼容性表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[PyTorch官方CUDA12.8 Torch索引](https://download.pytorch.org/whl/cu128/torch/)、[torchvision索引](https://download.pytorch.org/whl/cu128/torchvision/)。解析固定上述两个候选，其余依赖走PyPI，保存原索引和METADATA SHA。首次附加索引解析触发50MB解析目录保护线，进程已停止，未安装；失败目录保留，不用它冒充已通过依赖图。

先完成全部固定wheel的未压缩空间及逐包安装峰值估算，再根据重新测得的物理空间决定是否安装。环境仅放/home/scv/shared/phase111_ocm_runtime/下，逐包保存SHA和安装记录；可删除本轮成功安装后的受控wheel临时副本，不能清理历史结果。峰值估算含余量，不把WSL内部free当宿主free。固定环境之外的原cloud-lite-pt210只读；最终依赖检查和实际导入验收必须通过，不得用跳过依赖检查将旧Torch冒充新runtime。

本补充是执行授权与环境准备，尚未表示新环境安装或完整推理通过。后续失败保留成本/日志停止，不自动换场景、精度、patch、overlap或单模型继续。
