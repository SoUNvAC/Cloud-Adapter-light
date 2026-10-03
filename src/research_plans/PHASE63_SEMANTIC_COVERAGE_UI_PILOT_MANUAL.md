# Phase 63 Semantic-Coverage Exact-Pixel UI Pilot 手册

版本：1.0。本20项pilot只检验精确像元UI与标注流程，不进入类别准确率、模型、风险分数或正式confirmation统计。

## 抽样盲态

原始标签只在sealed端用于保证名义语义覆盖和边界配额，不向reviewer展示，也不作为正确答案。每个坐标由冻结规则从合格像元中确定，不存在实验者希望得到的类别。禁止根据附近最显眼对象、角标以前的位置或对实验意图的猜测作答。

## 唯一评价单位

只判断标题中`(x,y)`对应、由空心方框包围的一个源像元。上排192×192图仅提供上下文；下排及单波段图为21×21最近邻放大，每个色块就是一个源像元。十字线停在方框外，中心像元内部保持可见。附近云体或阴影只能作为几何和光谱上下文，不能代替中心像元成为标注对象。

## 固定填写顺序

1. `target_locatable`：能明确找到唯一中心像元填`yes`；坐标、页面或准星无法对应时填`no`，在notes说明，其余判断留空。
2. `identifiability`：位置明确后选择且只能选择一个：
   - `fine_classifiable`：中心像元具备细粒度语义判读条件；
   - `mixed_boundary`：中心像元位于类别、配准或混合像元边界；
   - `insufficient_evidence`：位置清楚但多光谱证据不足；
   - `unobservable_nodata`：纯色、填充值、nodata或无有效信息。
3. 仅当`identifiability=fine_classifiable`时填写一个`semantic_label`：`clear`、`thin_cloud`、`thick_cloud`、`haze_cirrus`、`cloud_shadow`或`terrain_water_shadow`。若中心像元确实位于thin/thick、cloud/shadow或clear/shadow边界，应填`mixed_boundary`，不得猜抽样者想要哪一侧。
4. `elapsed_seconds`记录首次打开到提交的秒数；`notes`只记录证据或技术问题。

## 独立性

A/B分别完成全部20项，每项只看一轮。在两份原始CSV通过格式校验、计算SHA256并冻结前，不得讨论pilot项目，不得查看对方判断、sealed manifest、原标签、模型预测、风险分数或抽样层。

## 预注册UI门

- A、B目标可定位率均至少95%，即各至少19/20；
- 坐标、页面和提交格式错误为0；
- 在双方均可定位项上，可辨识性exact agreement至少70%；
- 在名义interior项中，双方均判`fine_classifiable`至少10项；
- 上述双方均判可细分interior项的语义exact agreement至少60%。

只报告各名义抽样层的原始计数，不报告每类准确率。只有A/B原始判断冻结后才可解封名义层进行描述性比较，且名称固定为`reviewer–nominal-label concordance`。名义标签不是答案。本pilot通过也只允许重新预注册正式confirmation，不能证明标签可复现或模型有效。
