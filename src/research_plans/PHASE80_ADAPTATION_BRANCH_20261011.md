# Phase80：适配方法×光谱补充交叉探索

## 做什么

关闭Phase78/79固定方向增量分枝，不改分箱、方向或距离。转向模型分枝：比较另一种适配方法LoRA与现有MsRE，判断云影排序/损伤及光谱增量是否依赖当前适配方案。

本轮最多一个4000步LoRA作业、两个浅层拟合；不重训Source/MsRE，不加新特征，不做独立确认。本文件仅立项，尚未启动训练。

## 数据

- 原Catalogue fit的153组155产品用于LoRA监督适配；原153个fit代表用于浅层拟合与校准。32个development代表用于选checkpoint和评价，云影支持7组。不能把训练155产品与统计153代表混用。
- 原产品、mask、RGB处理、512裁块、增强和评估流程逐项复用Phase65a；不修正已用数据的重采样、不因Phase77未闭合另换输入。
- 不读Phase66、65b、65c或旧sealed像元/分数，不加入ALCD、不取新数据。已经用过的development仅提供探索证据。

## 1. 固定LoRA训练

1. 复用已有`Phase64LoRACloudAdapterDinoVisionTransformer`，rank=4、alpha=4、dropout=0，目标qkv/proj/fc1/fc2。使用原三父类Source checkpoint，SHA=`64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9`。
2. Source骨干基础权重、Cloud-Adapter和分割头全部冻结，只训练LoRA A/B。核验转换后的基础权重逐张量一致、未加载键仅限LoRA新增项，禁止用strict=False吞掉来源不明的缺键。
3. 预检查：零初始化残差时，同一fit输入LoRA关闭/开启均应复现Source logits，绝对差≤1e-6；确认可训练参数仅LoRA、梯度有限。失败先排查，不直接开始训练。
4. seed65、4000步、batch1、FP32、AdamW lr=1e-4/weight_decay=0.05，学习率调度及其他设置原样继承Phase65a。不搜索rank/lr/注入层或增加seed。启动前检查现有GPU任务，一次一个训练作业。
5. 每500步只用development mIoU选模型，平手取最早；沿用MsRE原选择目标，不能用本轮AP改选。保存全部8个checkpoint（到4000步），本轮仅最佳者参与正式探索对照，不事后比较其余节点。
6. 记录有效标签预算、实际参数量、训练时间和峰值显存。LoRA与MsRE参数量不完全相等；现有MsRE含target_head_delta而此LoRA不含，结果比较两套完整适配方案，不能归因于单一结构因素。

## 2. 固定四行主要对照

最佳LoRA在153 fit代表及32 development代表各前向一次，保存Shadow分数、三类预测、非Shadow胜者和SHA。其他模型复用缓存。

| 策略 | 浅层输入 | 非Shadow输出 |
|---|---|---|
| M | 原MsRE logit单输入校准（复用L1） | 原MsRE Surface/Cloud胜者 |
| M+S | 原MsRE logit＋B08/B11/B12（复用Phase67） | 同上 |
| L | LoRA logit单输入校准 | LoRA Surface/Cloud胜者 |
| L+S | LoRA logit＋B08/B11/B12 | 同上 |

原Source、原MsRE、LoRA原argmax和L3可作为附加参考行，不用它们替代主要比较。不同网络非Shadow输出不同，mIoU差不能全归因于Shadow分数。

只新增L、L+S两个逻辑回归：原fit索引/组等权BCE/0.001正则/优化器不变；网络分数系数非负，波段系数自由，标准化仅fit。各自用全部fit负类冻结全组平均FPR≤1%的exact-ties全局阈值。模型与阈值落锁后才评价development浅层结果。拟合不收敛不改参数补跑。

## 3. 固定比较与报告

- `L−M`：比较校准后的适配方案；`L+S−L`和`M+S−M`：各自光谱增量；`L+S−M+S`：补充相同光谱后的方案差。另列两光谱增量之差，不作纯机制因果结论。
- 四行完整报告宏/池化AP、冻结阈值召回、全部组及有/无Shadow组FPR、单侧95%FPR上界、逐组/池化mIoU、两种漏影流向、零召回及最差损伤。列全部7组原始AP差、支持分母和32组误报。
- 差值配对相关组bootstrap10000次、seed80011，只作条件于一次训练/既有开发选择的描述性区间。不能以多个checkpoint或像元数扩大独立样本数。
- 先复现M与M+S的原结果；异常停止排查。不删除失败景、不改阈值救指标。

## 如何收口

- 若L+S优于M+S且FPR/召回没有相反代价，保留LoRA分枝供下一轮必要消融；不宣称已独立胜出。
- 若两个适配方案都获得类似光谱增量，只支持“光谱补充可能不依赖MsRE”的探索解释，降低MsRE专属创新主张。
- 若LoRA没有改善或只改变工作点，关闭本轮固定LoRA方案，不搜索rank/lr。
- 所有结果均交付，无论正负不在本轮增加第三种网络。后续晋级与独立确认另定协议；当前全部旧确认池已参与研究，不能再作新方法的未见确认。

## 交付与停止

新目录`outputs/phase80/phase80_20261011/`，work_dir使用全新Phase80路径，防覆盖。交付短README、四行及附加参考表、逐组CSV、配置/参数/阈值/样本/输入SHA锁、训练曲线与检查点清单、日志。

代码本地验证后push、远端pull执行；不改无关文件、不自动同步work_dirs或清理历史数据。预算截至2026-10-20北京时间，不在到期后启动或延续本轮新研究训练。资源不足则记录未执行，不擅自缩训练步数或改方法。
