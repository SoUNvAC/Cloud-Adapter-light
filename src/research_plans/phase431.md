# Phase431：融合版加入输出特征残差层

- 做什么：Phase429融合版增加一个OutputFeatureResidual层，包含四路独立384→8→384残差，共26,144参数。在四路特征归一化后、金字塔缩放前执行；末层零初始化。
- 用什么：同一DINOv3-S+/16预训练、CloudSEN RGB三分类、同一分割头。
- 怎么做：从预训练重新跑40k，DINO冻结，其余联合训练；seed42901、batch2、AdamW 1e-4、500步预热、每2k验证。保留Phase429共同初始化。按用户要求与Phase430并行；先10步检查，再全新40k。
- 提交什么：源验证best/final、完整权重、更新检查；源best自动评测Catalogue32，比较Phase429融合版。
