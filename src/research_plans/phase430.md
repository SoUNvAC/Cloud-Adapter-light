# Phase430：原版DINOv2 CloudAdapter＋MsRE

- 做什么：原CloudAdapter骨干和原SparseMsREResidual，四处注入；不含局部读写、语义回写或输出残差头。
- 用什么：DINOv2-S预训练权重，CloudSEN RGB三分类，原轻量Mask2Former头。
- 怎么做：DINO冻结，其余联合训练40k；seed42901、batch2、AdamW 1e-4、500步预热、每2k源验证。任务模块重新初始化，不加载旧Source。核验预训练加载、优化器范围、各分支更新和冻结状态。
- 提交什么：best/final源指标、权重和训练记录；源验证best直接推理Catalogue固定32组。此轮为原版结构基准，不把跨骨干差异单独归因于DINO版本。
