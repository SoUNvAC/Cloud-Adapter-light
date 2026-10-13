# 连续Shadow分数：代码依据与未解除项

固定作者代码f68cf49b4cecc93ec5cb71e80110d76f61df073c；追溯其历史V1时期1c9b086828f64f78907ff7fcc265003c8918ff4c（2024-09-25）。后者证明旧路径，不宣称作者论文确切执行commit。函数行号与源码SHA见CODE_EVIDENCE.json；sources/为小型作者代码快照，不含权重。

## 输入到输出

1. 当前及历史data_loaders.load_s2默认 **[B04,B03,B8A]**，不是B08；resolution范围10—50m。open_s2_bands读取L1C/L2A DN，尺度不同用bilinear，并更新原点/transform；没有Source的*255、BGR转换、512 resize或全局标准化。现有Phase104 B08六波段不能据此称严格论文输入等价。
2. predict_from_array要求(3,H,W)。当前check_patch_size（248—307）硬最小32、建议50；默认patch1000、overlap300，显著nodata或图小会调整。make_patch_indexes（raster_utils:111—144）stride=patch-overlap，末端窗口回贴，不自动给全景补边；重复终端窗口被批编译去重。get_patch会移动窗口减少nodata；无效窗口跳过。padding仅是小于最小尺寸时给调用方的建议，不能当已冻结论文规则。
3. get_patch→channel_norm（model_utils:78—102）：每块每通道只对非nodata位置取均值和总体std，std=0改1，nodata填0。正仿射量纲在有效支持不变时被归一化消除；这不能证明任意offset/无效掩膜或不同上下文等价。NaN/SATURATED必须在后续输入验收中显式处理；不能只借Source掩膜当作者已经处理过。
4. get_models(version=1)选择索引两行；collect_models加载全部两模型。build_fastai_model为3输入/4输出DynamicUnet，timm pretrained=False。infer_batch（model_utils:170—180）是**先平均两个model(...)输出**，不是每个模型softmax后平均；没有依据可擅自交换顺序。外部fastai/timm的实际输出语义/依赖版本仍待真实环境验收，代码将其作为logit式类分数处理。
5. store_results（137—167）先乘二维类别共用gradient，再累积。gradient从重叠边缘1/overlap线性升高并相乘；coordinator（366—468）在confidence分支除以累计gradient，之后才调用postprocess_preds。
6. postprocess_preds（310—330）：export_confidence=True, softmax_output=True时，在类维softmax，然后**+0.001并截断[0.001,0.999]**；NaN变0。False confidence分支才argmax。取第3通道是作者定义的Shadow confidence；数值在有效处0.001—0.999，无效可为0。由于偏移/截断，四通道通常不和为1；这是作者softmax衍生confidence，**不是经校准正确概率**。截断引入的并列必须保留，不能另用未截断softmax或平均概率替换以改善AP。

## 现成接口与最小包装方案（未调用）

已有export_confidence接口，无需新增网络forward hook。计划显式固定model_version=1、两模型、compile_models=False、float32、patch/overlap、workspace内模型目录与冻结HF revision；取confidence[3]，另外输出valid/coverage。不能调用默认None版本或下载器默认应用数据目录。现成下载器没有revision/checksum参数且可能自动下载，应在另行授权后先由受控取得/校验步骤固定文件，再允许推理；本轮完全没有调用collect_models/get_models。

valid应将作者nodata规则、实际预测覆盖、全部所需输入有限且非special值、冻结比较支持分别保存并取交集；未覆盖位置不得当均匀0.25或可用0分数。完整场景执行需保留作者patch上下文；原Source1020→512窗口和点缓存不能直接当OCM输入算公平复现。当前作者nodata有效条件为至少一通道非nodata（raster_utils:75—95）；历史V1为全部通道非nodata（旧raster_utils:58—64）。历史get_patch与当前实现也变动。必须明确选择/冻结所复现代码语义，不能把当前兼容V1权重称论文逐像元完全复现。

## 准入边界

**静态连续接口有代码依据；真实AP运行准入尚未通过。** 原文档soft voting不作为任意概率平均授权，以已追溯代码先平均类输出的路径为准。14项纯数组检查包括实际作者NumPy函数AST抽取、拼接代数和模型索引；没有torch、权重或真实输出。它们不验证fastai输出、真实网络一致性、概率校准、上下文质量或运行成本。
