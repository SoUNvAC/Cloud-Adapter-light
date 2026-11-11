# Phase213：源域损失预算选择插值强度，并在目标测试集确认

## 1．执行目标

验证Phase212-R的权重插值是否能变成一个不查看目标测试标签的适配规则：

> 用源域验证集预先规定可接受的源能力损失，选择最大的插值强度α，再一次性评价未参与选模的目标test。

本轮只检验固定解码头插值，不提出门控网络、不训练新模块、不做标签预算曲线、不重训LoRA／MsRE／head-only。Phase212-R的重训端点和三个插值头直接复用；如文件无法按锁恢复，停止并交缺项，不能偷偷重训或换checkpoint。

## 2．固定模型与候选α

每个目标域分别使用Phase212-R的Source head与该目标head：

```text
head(alpha) = head_source + alpha * (head_target - head_source)
```

候选固定为：`α ∈ {0, 0.25, 0.50, 0.75, 1.00}`。已有的0.25／0.50／0.75头可复用；α=0和α=1端点必须由锁定权重重建并核对。不得搜索更密α网格、不得根据target-test结果添加候选。

目标域：L8、SPARCS各自一套目标head。完整DINOv2／Cloud-Adapter backbone、非head参数、buffer及输入处理全部冻结、eval。插值只作用于完整decode_head浮点参数；非浮点buffer按锁定规则保留。每个α独立保存head SHA和参数验收。

## 3．源域预算选择规则

源域评价使用CloudSEN12 High原官方val，按Phase212-R同一RGB／512／三父类／FP32管线，源域按既有107个ROI分组统计。CloudSEN test始终封存。

源域基准为Phase213重新核验的α=0 Source结果。分别预注册三种源损失预算：

```text
B1：源域池化mIoU损失 ≤ 1个百分点
B3：源域池化mIoU损失 ≤ 3个百分点
B5：源域池化mIoU损失 ≤ 5个百分点
```

对每个目标域、每个预算B，选择满足下式的**最大α**：

```text
source_mIoU(alpha) >= source_mIoU(alpha=0) - B
```

选择只读取源域val标签；不能读取L8／SPARCS target-test标签、指标或错误流。若多个α满足，选最大的α；若因数值并列，选较小计算成本相同的最小索引并记录。若无α满足，报告无可行点。预算选择结果一旦封存，不得因target-test表现修改。

注意：B1／B3／B5是三种预设工作点，不是三个独立方法，也不允许事后挑一个最漂亮预算作为唯一结论。完整报告三条预算线。

## 4．评价划分与数据边界

### 源域

- CloudSEN12 High官方val，535张、107个ROI，作为源能力保持评价和α选择集。
- 不读取官方test，不新增源域数据。
- 源域只用于选择α和报告保持代价，不把源val结果称独立确认。

### L8目标域

- 只读取原Phase64 scene-disjoint manifest中的target-test。
- target-train与target-val不重新读取用于本轮选择；Phase212-R端点已经按原target-val选定，历史暴露关系必须保留。
- 目标test标签只在全部α和预算选择锁定后读取。

### SPARCS目标域

- 只读取原Phase64 scene-disjoint manifest中的target-test。
- 不读取target-train或target-val作本轮选择。
- 若原SPARCS test清单、类别映射或输入文件身份无法核验，停止该域并报告缺项，不临时改用val。

不要把L8／SPARCS target-test称为与VFM预训练完全独立；VFM完整训练覆盖未知。它是**同一数据协议下未参与本轮模型／α选择的目标测试集**。

## 5．评价输出

每个目标域固定报告五个候选α，以及B1／B3／B5各自选中的α。

主表：

| 目标域 | 预算 | 选中α | 源val池化mIoU | 源损失 | target-test池化mIoU | target-test Shadow IoU | target-test Cloud IoU | target-test Surface IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

完整候选表：

| 目标域 | α | 源val池化mIoU | 源逐ROI平均mIoU | 源Shadow IoU | target-test池化mIoU | target-test逐scene平均mIoU | target-test各类IoU | Shadow召回 | 非Shadow→Shadow误报 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|

另报：完整混淆矩阵、逐ROI／逐scene结果、错误流、有效像元、ignore数量、缺类情况和零召回scene。池化指标由汇总混淆矩阵计算；逐组mIoU不代替池化mIoU。

历史参照只引用Phase64最终审计与Phase212-R已有结果：Source、shared-parent、MsRE、LoRA。不得把历史FP16诊断值与本轮FP32主结果混成严格比较；若历史精确权重／支持无法重现，列为描述性参考。

## 6．主要判断

本轮只回答以下问题：

1. 源域预算选择出的α，在未参与选择的target-test上是否仍保留目标收益？
2. B1、B3、B5三条预算线是否产生稳定、单调且可解释的目标—源域取舍？
3. L8与SPARCS是否呈现相同规律？
4. 选中的插值点是否至少在点估计上超过Phase64历史MsRE坐标？

不能据本轮宣称：

- 插值算法本身是新方法；
- 已证明通用DINO能力保持；
- 目标test完全独立于Omni／VFM训练；
- α规则在其他任务或传感器上普遍有效。

## 7．统计与正确性

- 源域val按107个ROI重采样；L8／SPARCS target-test按原scene重采样。分别对每个固定α报告池化mIoU差的10000次配对组bootstrap，seed `213020`。
- 目标test不得用于选择α、阈值或checkpoint。所有α必须在target-test读取前完成头部SHA、候选表和预算选择锁。
- 重新核验α=0在源val及两个目标test上的Source结果；若与Phase212-R主结果差异超过原报告精度容忍范围，先查输入／精度／映射，不覆盖旧结果。
- 独立CPU复核插值公式、参数形状、非head冻结、类别映射、ignore分母、混淆矩阵和预算选择方向。不得只用同一主脚本复核。
- 源mIoU预算是池化mIoU预算；逐ROI损失可单列，不能把最坏ROI损失隐藏在池化均值中。

## 8．资源和执行限制

- 零训练、零优化器更新、零新权重／影像下载、零环境升级。
- 预计为已有端点和五个α的源val、L8-test、SPARCS-test固定前向；执行前按实际manifest计算调用数。总新增网络调用≤12000次，GPU阶段≤3小时，GPU≤20GiB、CPU≤16GiB、新持久文件≤4GB、宿主物理盘余量≥4GB。
- 不读取CloudSEN test、其他确认池、Omni新数据或历史封存外部池。
- 不自动push、部署或开始下一phase。遇到缺失checkpoint、输入身份或test映射问题，保留已完成锁和缺口，不换数据、不缩场景凑预算。

## 9．提交结果

输出：`outputs/phase213/phase213_20261021/`；摘要：`src/research_results/phase213_20261021/`。

必须提交：

- `RESULT.md`：B1／B3／B5三条预算线、五个α候选、L8／SPARCS target-test结果和限制；
- `ALPHA_SELECTION_LOCK.json`：源val结果、预算、选中α、候选顺序、SHA；
- `PARETO_TEST_TABLE.csv`：所有候选及历史参照；
- 完整逐ROI／逐scene表、混淆矩阵、错误流、bootstrap区间；
- Source／目标head、代码、输入manifest、标签映射、插值参数和评价结果SHA；
- 独立CPU验收、真实调用／GPU／存储账本、失败与缺项。

README最后只给三种状态：

1. 源预算规则在两个target-test均保留有用目标收益；
2. 只在一个target-test成立；
3. target-test不支持稳定收益。

不因结果不理想追加α、重训、门控或正则化；门控与标签预算曲线另行立项。
