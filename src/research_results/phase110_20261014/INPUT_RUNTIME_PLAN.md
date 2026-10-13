# 作者默认B8A、锁定代码V1的旧池兼容探索方案（不执行）

## 输入身份与共同支持

固定原Phase104 16产品，原N02.04/02.05/02.06，B8A每对象name/generation/长度/镜像声明MD5与旧listing一致；CRC32C本次由对象元数据补记。原产品XML PRODUCT_URI、IMAGE_FILE、bandId8/physicalBand B8A/RESOLUTION20和原tile20m格网均校验。GCS是镜像提供者checksum声明，不是本地内容校验；未下载波段，未核实JPEG2000像元可读性。旧Phase104/105输入和结果完全保留。

后续OCM输入按[B04,B03,B8A]原产品DN及作者bilinear到20m，使用作者逐块非nodata归一化；不套Source2×2均值、BGR、×255或512缩放。两方法可保留各自作者算子，但比较位置/参考语义/有效支持必须相同。将原坐标映射到同一原20m格网，每点取包含格分数。B8A/special值/输出覆盖新增无效点必须在评价前冻结共同交集并报告，不能直接把旧G7数值与新不同支持OCM表拼接。

## 运行路径与接口草案

仅在后续授权与依赖验收后，计划在/home/scv/shared/phase110_ocm_v1/下分weights/<HF revision>、b8a/<原产品>、runtime、temp、scores目录；本轮没有创建这些远端目录。原six-band JP2复用，不复制全部影像，不访问work_dirs。

- 明确固定HF revision、双SHA、两架构、n_out4、in_chans3、model_library=fastai；分别用固定作者load_model_from_weights加载.safetensors并eval。源代码已支持此格式，无需默认下载器转换.pth；真实state兼容性仍需后续验收。
- 向固定predict_from_array传两项custom_models、float32、patch1000/overlap300、batch1、CPU mosaic、export_confidence=True、softmax_output=True、compile_models=False。不调用get_models/get_model_data_dir，禁止自动latest、隐式下载/用户缓存、编译预热。
- 在完整原20m场景上保留作者滑窗、边缘回贴、nodata移动、两模型输出均值、gradient累积与归一及softmax+0.001/clip。不得为了省空间只跑Source窗口或点周围小裁块。当前作者代码的无效规则应明确冻结为当前版本，不把它称历史论文逐项一致。
- 每景只从最终四通道confidence提取第3通道固定点位分数，并保存valid/coverage及可复核处理日志，随后释放密集内存。不把截断confidence称校准概率，不改概率平均顺序，不保存16景全部密集输出。

## 实施前门与停止

现有Python3.10满足当前声明，但Torch2.1低于>=2.3，缺fastai/smp/gdown；不能直接准入，也不在原cloud-lite-pt210安装或升级。独立runtime、依赖版本/包清单/许可/峰值与真实权重载入须后续另行授权验收；不得换历史版本以绕过当前声明。硬件容量只支持规划，不证明运行一致性/速度。若峰值/宿主保留量/版本/输入/权重校验失败，停止并保留日志，不改patch、换B08、换权重或扩大预算。

旧池只能探索，不恢复严格论文复现或新确认。是否真的取权重/B8A、创建runtime和推理，需要下一执行协议；Phase110不自动进行。
