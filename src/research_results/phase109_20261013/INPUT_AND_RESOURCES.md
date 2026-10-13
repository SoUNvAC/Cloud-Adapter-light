# 输入与资源条件（仅元数据）

**输入。** Phase104已有16产品每景B02/B03/B04/B08/B11/B12；只读取object_manifest与input_operator_lock，未读栅格或标签。20m在OCM声明10—50m内，但作者S2默认B04/B03/B8A；B08虽属于一般NIR输入，尚不能当默认S2/V1论文配置等价。Phase104 native10→20使用对齐2×2均值，作者load_s2使用bilinear；Source的BGR、*255、1020→512均不适用OCM。严格V1比较的波段/重采样/上下文条件未解除，不在本轮补B8A或悄然改用B08。

**权重。** V1两份safetensors公开声明72,314,968和130,164,944 bytes，合计202,479,912 bytes（约202.48MB）。原Google Drive .pth的长度/checksum unknown；作者当前下载器将HF safetensors转换为.pth。HF revision和LFS SHA已锁，但没有下载验证内容、tensor身份或与原Google包逐字节相同。无需解压这两份非归档权重；转换.pth、HF双缓存、临时文件实际峰值未实测，不能把202MB等同总资源。

**4GB门。** 仅声明权重下载量<4,000,000,000 bytes已核验；全部新增中间文件峰值≤4GB仍pending。以常见5490×5490四类float32为例，单景四通道文件482,241,600 bytes；若16景全部保存约7.716GB，已超过建议上限。只保留必要分数/点索引并逐景释放是后续方案，不能在本轮执行。输入/gradient/模型激活主要占RAM/VRAM，不能从下载大小推耗时或可用显存；也不拿过去宿主余量代替当前检查。

**依赖。** 固定当前pyproject要求Python>=3.10、torch>=2.3、V1需legacy fastai>=2.7及timm等。真实现有cloud-lite-pt210能否满足未验收；本轮未安装、导入模型包、改环境或编译预热。代码compile_models会额外前向，后续若允许也必须保持False并记录。下载器默认用户应用缓存不在授权workspace，不能直接调用默认行为。
