# Phase 63 精确像元定位 UI Pilot 操作手册

版本：1.0。该pilot只验证标注界面与判读流程，不进入模型、风险分数或最终confirmation统计。

## 唯一评价对象

每个项目只判断标题给出的`(x, y)`中心像元。上排192×192图只提供场景上下文，黄色角标仅表示目标位置；禁止按角标附近或整幅图面积最大的类别作答。下排及单波段图是21×21源像元的最近邻放大，空心方框严格包围一个源像元，十字线停在方框外。方框内部不得被标记遮挡。

## 固定填写顺序

1. `target_locatable`：能够明确找到空心方框包围的唯一像元时填`yes`，否则填`no`并在notes说明；此时其余判断留空。
2. `identifiability`：位置明确后，只能选择：
   - `fine_classifiable`：中心像元具备细粒度判读条件；
   - `mixed_boundary`：中心像元位于类别、配准或混合像元边界；
   - `insufficient_evidence`：位置清楚但多光谱证据不足；
   - `unobservable_nodata`：纯色、填充值、nodata或无有效信息。
3. 只有`identifiability=fine_classifiable`时填写`semantic_label`：`clear`、`thin_cloud`、`thick_cloud`、`haze_cirrus`、`cloud_shadow`或`terrain_water_shadow`。不得给混合边界强加语义类别。
4. `elapsed_seconds`记录该项目从首次打开到提交的总秒数；`notes`只记录证据或技术问题，不猜原标签和模型结果。

## 盲态和独立性

Reviewer不得查看原标签、模型预测、风险分数、抽样来源、sealed manifest或对方判断。两人独立完成全部16项，每项只看一轮；在两份原始CSV冻结前不得讨论pilot项目。允许使用全部B1–B11、推荐合成、太阳方位与必要元数据。

## 预注册读出

- 两位reviewer各自的目标不可定位次数与耗时分布；
- 在双方均可定位项目上的可辨识性exact agreement、Cohen κ与Gwet AC1；
- 仅在双方都选`fine_classifiable`的项目上计算语义exact agreement、κ/AC1与类别分布；
- 描述`mixed_boundary`、`insufficient_evidence`和`unobservable_nodata`的分歧，不把它们与语义类别放进同一竞争标签空间。

UI可行性门固定为：双方目标可定位率均至少95%；可辨识性exact agreement至少70%且AC1至少0.40；双方均认为可细分的项目至少6个，且这些项目的语义exact agreement至少50%。样本量仅16，结果只能决定是否值得重做正式confirmation，不能证明界面改动导致一致率提升，也不能替代Phase63最终生死门。
