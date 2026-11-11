# Phase429：DINOv3-S+/16 两版对照

- 做什么：A=冻结DINOv3＋CNN多尺度上下文＋四层交叉注意力；B=A＋MsRE＋局部细节读取和语义回写。实现独立于旧CloudAdapter代码；A是原思路的最小重实现，不声称逐算子复现。
- 用什么：官方DINOv3-S+/16通用预训练权重；同一个轻量Mask2Former头，CloudSEN RGB三分类数据。
- 怎么做：两版同seed42901、共同模块及头初始化一致、batch2、FP32、40k步、500步预热、AdamW 1e-4、PolyLR归零；每2k验证。DINO冻结，其他模块联合训练。先验证官方前向一致性、特殊token、梯度、优化器范围，再正式跑。
- 比什么：源验证集best/final及三类IoU；两版各自源验证best在Catalogue固定32组development上的直接推理结果，作为主要目标域比较。不用目标集挑源检查点，不混入目标适配后的历史分数。
- 提交什么：两版权重、完整配置/代码/权重哈希、训练曲线、目标/源指标、训练时间和显存。单seed先判断方向，正向结果再补稳定性。

## 运行

在src目录设置 `DINOV3_REPO`、`DINOV3_WEIGHTS` 和可选 `CLOUDSEN_ROOT`：

```bash
python -m phase429.check
python -m phase429.train baseline --work-dir /home/scv/shared/phase429_20261102/baseline
python -m phase429.train fused --work-dir /home/scv/shared/phase429_20261102/fused
```

check须把官方DINOv3代码目录加入PYTHONPATH；只运行随机初始化结构检查，不代表性能验证。正式入口缺少预训练权重会报错，禁止静默回退随机主干。Catalogue评价入口后续接入现有固定协议，本轮先交付网络与训练入口。

## 已准备的远端位置

- 官方源码：`/home/scv/shared/phase429_20261102/dinov3`，commit `6876159a11b4df116f30f667f8c9888617df0751`。
- 新网络：`/home/scv/shared/phase429_20261102/code/phase429`。
- 旧PyTorch2.1仅需对源码的编译缓存配置加存在性判断，脚本为 `python -m phase429.compat "$DINOV3_REPO"`。未改DINO模型运算，未升级旧环境；训练入口记录源码diff哈希。
- 权重：用户提供 `D:/master/model.safetensors`，已传到远端同根目录；SHA256 `208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2`。`convert_weights.py` 严格转换为 `dinov3_splus_native.pth`，全部参数加载，没有随机补齐。
- 启动：两版先各做10步真实数据检查；通过后 `python -m phase429.queue --root /home/scv/shared/phase429_20261102` 依次执行两个40k。正式训练不接续检查权重。进度见远端 `QUEUE_STATUS.json` 和 `baseline.log` / `fused.log`。
