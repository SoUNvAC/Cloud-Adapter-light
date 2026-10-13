# Phase109：OmniCloudMask V1强对照准入核验

## 1 数据结果

已核验Phase108材料SHA，固定当前作者commit并追溯V1历史；两模型完整集成身份一致。公开HF V1权重声明总量202.48MB、各SHA256及MIT模型卡已记录，未下载权重。源码追踪确认已有连续Shadow confidence接口；14项人工合成数组/索引检查通过。

本轮训练/拟合/网络前向/性能评价/影像与权重下载/环境安装全部0。只读公开代码与元数据；访问body约0.413MB，见access_ledger.json，未读取栅格/标签或运行模型。协议目录沿用20261013，实际完成2026-10-14。

## 2 结论分析

**完成准入材料核验；尚不准入真实强对照运行。** 静态分数接口可追溯，不能称已校准概率。作者S2默认[B04,B03,B8A]，现有B08不是已经证明等价的论文输入；当前与历史无效掩膜也有差异。V1训练完整清单、真实依赖/输出一致性和中间空间峰值未验收。新独立密集参考条件仍未满足。

文件：VERSION_SOURCE_LOCK.json、SCORE_INTERFACE.md、WEIGHTS_AND_LICENSE.json、TRAINING_OVERLAP.csv、OVERLAP_RULES.json、INPUT_AND_RESOURCES.md、ARRAY_CHECKS.json、ADMISSION.md、CODE_EVIDENCE.json、SOURCE_SHA256.json及项目输入SHA。小型作者代码快照含MIT LICENSE，来源URL见访问清单；不包含论文全文或模型。

实际可行下一步仅为补齐作者V1清单/配置及输入、缓存、环境方案；条件解除后另行授权。没有把原先“权重可能很大”的unknown伪装为阻塞：已知下载量较小，主要缺口是输入/谱系/参考与实际资源验收。
