# Phase400：拆开 Cloud-Adapter＋MsRE，看收益来自哪里

## 1. 做什么

固定现有 RGB Source，训练三个模型，判断哪部分值得继续优化：

| 模型 | MsRE | 输出残差头 |
|---|---|---|
| M+H：当前完整组合 | 开 | 开 |
| M：仅 MsRE | 开 | 关 |
| H：仅输出残差头 | 关 | 开 |

另用未适配的 Source 作基线，不重训 Source。本轮不增加新模块。

## 2. 用什么做

- 代码：`src/cloud_adapter/models/backbones/target_msre_dinov2.py`。
- 参考配置：`src/configs/protocol/phase65a_explore_rgb.py`。
- Source：Phase64 的 `best_mIoU_iter_4000.pth`，SHA256 为 `64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。
- 数据：Catalogue 原153个fit代表用于训练、32个development代表用于评价；CloudSEN val535用于检查源域表现。沿用原标签、输入处理和有效像元，不读test。
- 环境：优先复用远端 `cloud-lite-pt210` 和单张4090D 24GB。开始前确认权重、数据、环境都在。

## 3. 怎么做

1. 给 MsRE 和输出残差头增加独立开关，保持旧配置行为不变；所有结果写入新的Phase400目录。
2. 检查权重加载、可训练参数和梯度。只训练开启的分支，Source参数及BN等状态保持不变；关闭全部分支时应恢复Source输出。共同模块使用相同初始权重。
3. 每个模型先跑20步预热＋100步测速，检查loss、显存和耗时。预计总GPU时间按12小时规划（含2小时恢复余量）；放不下就先调整计划，不盲目开长任务。
4. 三个模型依次训练：seed40001、batch1、4000步，优化器、学习率和增强沿用参考配置。短跑后重新初始化正式训练；每500步评价，按development三类池化mIoU选最佳，同时保留最终结果。
5. 评价最佳和最终模型：目标域报告三类mIoU、云影宏AP、Shadow IoU；源域报告三类mIoU。记录训练/评价耗时、峰值显存和参数量。相同checkpoint不重复评价。

**注意：源域评价必须开启该模型训练过的分支。** 历史“源域只下降0.04”是在关闭MsRE分支后测得，不能作为开启适配后的抗遗忘证据。关闭分支只单独做一致性检查。

遇到NaN、错误标签、冻结权重变化或OOM，停下定位后再恢复，不偷偷更改batch或精度。三个对照不因早期loss较差提前淘汰。

## 4. 做完交付什么

放到 `outputs/phase400/phase400_<运行日期>/`：

- `README.md`：一张对照表＋简短结论，说明收益来自哪里、下一步建议改哪里。单seed、已暴露development结果只作探索。
- `summary.csv`：四个模型的目标mIoU、云影宏AP、Shadow IoU、源mIoU、参数量、耗时、峰值显存；训练模型区分最佳/最终checkpoint。
- `per_group.csv`：目标域逐组结果，保留不利组。
- `run_info.json`：代码版本、实际配置、数据和Source哈希、seed、分支开关、checkpoint路径与哈希、完成或失败状态。
- 训练日志和可恢复checkpoint位置；大权重留训练机器，不必全部回传。

向 `src/EXPERIMENT_LOG.md` 追加简短结果。把交付目录和README路径发回本聊天，再决定Phase401做什么。

当前状态：计划已写，尚未开始实现或训练。
