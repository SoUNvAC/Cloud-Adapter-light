"""Bounded factual report from completed fixed-cache diagnostics."""
from phase308_common import *
def report():
    acceptance=load(ROOT/'CPU_ACCEPTANCE.json');details=load(ROOT/'DETAILED_RESULTS.json');audit=read(ROOT/'IMAGE_REFERENCE_AUDIT.csv')
    largest=next(r['product'] for r in audit if 'T25XEG' in r['product'])
    rows=[r for r in details if r['product']==largest];bounds=read(ROOT/'HELDOUT_BOUNDS.csv');scores=read(ROOT/'SCORE_DISTRIBUTIONS.csv');composition=read(ROOT/'TRAIN_COMPOSITION.csv')
    support=[r for r in bounds if r['product']==largest and r['model']=='MS6' and r['space']=='raw6' and r['heldout_label']=='D' and r['feature']=='ANY']
    snow=[r for r in composition if r['heldout']==largest and r['label']=='D' and int(r['n'])>0]
    snow1=[r for r in snow if r['snow_ice']=='1']
    lines=['# 数据结果','',f"全部32折×3模型复现；概率最大绝对差{acceptance['probability_max_error']:.3g}，贡献和误差{acceptance['contribution_sum_max_error']:.3g}。原修复/撤销、Brier、AUC全部一致；纯D的AUC仍NA，10个空C折保留。零拟合、零训练、零神经前向。",'',
        '|T25XEG固定模型|D/K|修复D|Brier|D的p(K) min/median/max|', '|---|---:|---:|---:|---|']
    for r in rows:lines.append(f"|{r['model']}|{r['D']}/{r['K']}|{r['repaired_D']}|{r['Brier']:.16g}|{r['p_D']['min']:.6g}/{r['p_D']['p50']:.6g}/{r['p_D']['max']:.6g}|")
    lines+=['',f"T25XEG留出折训练D来自{len(snow)}组；snow/ice=1的训练D来自{len(snow1)}组、{sum(int(r['n']) for r in snow1)}个原拟合样本。原等组权及条件类权均列TRAIN_COMPOSITION.csv。",'', '|T25XEG D vs 训练类（原六波段任一超界）|min/max外比例|p1/p99外比例|','|---|---:|---:|']
    for r in support:lines.append(f"|{r['training_label']}|{float(r['frac_minmax']):.6%}|{float(r['frac_p1p99']):.6%}|")
    lines+=['','固定前三组均通过产品、cube通道、flat-index、参考one-hot、缓存事件与来源SHA核对；全景和中心/max-D/max-K预览位置固定，无K裁剪显式NA。IMAGE_REFERENCE_AUDIT、PREVIEW_COORDINATES、DISPLAY_SATURATION记录原编码/类数/场景/坐标/显示截断。仅显示裁剪到[0,1]，模型输入未裁剪。','',
        '# 结论分析','',
        '1. 身份/实现：本轮未发现概率计算、采样权重、通道或参考对应异常；该范围内的检查不能覆盖原始发布前标注正确性。',
        f"2. 支持：留出最大组后仍有{len(snow)}个D训练组，其中雪冰场景D仅{len(snow1)}组。六波段D/K双方区间及重叠、逐维/任一超界在表中完整保留；这些是同类支持不足或外推线索，不能把单变量在范围内当作联合分布无偏移，也不能断言不可分。",
        '3. 分数迁移：最大组全部D在三种固定模型中均p(K)>=0.5，因此固定读出保留Cloud、修复为0；SCORE_DISTRIBUTIONS和SIGNED_CONTRIBUTIONS保存逐特征有符号均值/分位数与截距。训练内分数只作对照，不当泛化成绩；贡献相关性不解释为物理因果。纯D组没有组内AUC证据，也未证明调阈值/校准能够解决。',
        '4. 参考口径：最大组原共同支持全Surface及snow/ice标签按原记录保留；全Surface不自动错标，场景含雪冰不确定损伤像元真值。README在固定环境缺PDF解析器，保留SHA及内容读取缺口；record未明确备用mask编码/独立性，故alt_masks=unknown且未读，不替换主参考。图像核查不能证明云雪像元真值，结论为无法判定。',
        '', '本轮不改变Phase307“有限多光谱增量、最大损伤仍未解决”的边界。前三组属历史损伤排序的事后诊断，全32折保留；目标参考监督、反复开发暴露、单seed、Source/DINO预训练交集未知的限制仍在。没有重新拟合、调阈值、部署或启动下一phase。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    table(ROOT/'EXPLANATION_EVIDENCE.csv',[
        dict(explanation='identity_implementation',evidence='all96 original probabilities/counts/Brier/AUC and sample/weight SHA reproduced',counter_or_gap='pre-publication data correctness outside scope',coverage='32 folds; fixed3 original-image/reference checks',explains_largest='no concrete discrepancy found',next_change='none in Phase308'),
        dict(explanation='same_class_support',evidence=f'largest fold training D groups={len(snow)}, snow-D groups={len(snow1)}; full D/K intervals and bounds retained',counter_or_gap='univariate coverage cannot characterize joint support or establish causality',coverage='all nonempty fitted folds, each feature/class',explains_largest='support warning; insufficient for sole cause',next_change='future separately authorized design required'),
        dict(explanation='score_transfer',evidence='largest allD probabilities>=.5, signed sums independently match logits',counter_or_gap='pureD no ranking evidence; no threshold remedy established',coverage='all96 folds/models and detailed3',explains_largest='directly explains zero fixed-readout repair',next_change='no threshold or calibration change'),
        dict(explanation='reference_semantics',evidence='primary masks/source tags/cache identity consistent',counter_or_gap='visual cannot determine true cloud/snow; README parser unavailable; alt independence unknown',coverage='fixed3 only',explains_largest='unresolved semantic correctness; no evidence of actual mismatch',next_change='preserve reference and unknowns')])
