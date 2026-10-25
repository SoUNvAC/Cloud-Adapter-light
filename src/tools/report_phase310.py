"""Factual bounded report, fixed contrasts and all prespecified endpoints."""
from phase310_common import *
def report():
    metrics=read(ROOT/'METRICS.csv');damage=next(r for r in read(ROOT/'DAMAGE_BENEFIT.csv') if r['product']=='ALL' and r['region']=='all');curve=read(ROOT/'TRAINING_SUPPORT.csv')[0]
    lines=['# 数据结果','',f"一次全新seed30201、LoRA294912参数，1000尝试/955更新/45跳过；原A=75469158、三类40622307/33554491/1292360精确复现。P唯一产品像元{curve['P_unique_product_pixels']}，更新裁剪累计P{curve['P_cumulative_updated_crops']}；A/P不交且仅影像valid，原始样本/顺序/翻转不变。",'',f"lambda=1、T=1：CE(s,pseudo) mean_A + KL(softmax(Source_same512_s)||softmax(student_s)) mean_P；平均PL/SP/total={float(curve['mean_PL']):.8g}/{float(curve['mean_SP']):.8g}/{float(curve['mean_total']):.8g}。两网络独立、Source全冻结；原基础参数/buffer不变；零LoRA及原实现lambda0一步loss/梯度/更新门检见MODEL_ACCEPTANCE。",'', '|策略|组均/池化mIoU %|三类池化IoU %|Shadow宏/池化AP %|宏召回 %|宏/最大FPR %|','|---|---|---|---|---|---|']
    for r in metrics:
        f=lambda k:f"{float(r[k]):.4f}"
        lines.append('|'+ '|'.join([r['strategy'],f('group_mean_mIoU_pct')+'/'+f('pooled_mIoU_pct'),'/'.join(f(k) for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct')),f('macro_AP_pct')+'/'+f('pooled_AP_pct'),f('macro_recall_pct'),f('macro_FPR_pct')+'/'+f('max_group_FPR_pct')])+'|')
    lines+=['',f"全部32组保留；AP有效组数{','.join(r['strategy']+':'+r['AP_groups'] for r in metrics)}。缺类IoU按原union支持平均，零分母NA。固定末步评价，Source/Q90-PL缓存身份/支持/SHA复用；旧Q90首development9窗重放先通过<=1e-6及pred一致。",'',f"主要损伤终点：原D={damage['original_D']}，转Surface/仍Cloud/转Shadow={damage['D_to_Surface']}/{damage['D_still_Cloud']}/{damage['D_to_Shadow']}；原D外新增Surface→Cloud={damage['outside_D_new_Surface_to_Cloud']}。原纠正集合={damage['original_Cloud_correction']}，保留Cloud/退回Shadow/转Surface={damage['benefit_retained_Cloud']}/{damage['benefit_back_Shadow']}/{damage['benefit_to_Surface']}，保留率{100*float(damage['Cloud_correction_retention']):.3f}%；集合外新增SourceShadow→Cloud纠正{damage['outside_B_new_SourceShadow_to_Cloud']}。",'', '两种比较、P/A/其余/全部区域的32逐组及全池27格、六种有向错误的消除/新增/改正/转另一错误另表；保护区真Cloud/Shadow的误保护代价见PROTECTION_COST。非G D=450058仅作历史口径复现，不是训练P或总D分母。4探针×4时点×区域×真值均交，Q90按匹配尝试步对照；不选模。', '', '# 结论分析','']
    a=next(r for r in metrics if r['strategy']=='Q90-SP');b=next(r for r in metrics if r['strategy']=='Q90-PL')
    delta=float(a['group_mean_mIoU_pct'])-float(b['group_mean_mIoU_pct'])
    lines.append(f"主比较Q90-SP−Q90-PL组均mIoU {delta:+.6f}pp。修复原D与原D外新增损伤必须同时看；保留纠正不能用错误净减少数代替。固定约束的结果只说明本次lambda=1、单seed和本协议的实际取舍，不证明所有保持约束有效或无效，也不证明损伤全部源于未接纳区域。")
    worst=[r for r in read(ROOT/'PER_GROUP_METRICS.csv') if 'T25XEG' in r['product']]
    lines.append('最大历史损伤组T25XEG：'+'；'.join(r['strategy']+' mIoU='+r['mIoU_pct']+'%、union类数='+r['union_classes'] for r in worst)+'。固定前三组诊断保留，不按结果删组或调lambda。')
    lines+=['目标训练入口不读参考、场景标签或D/K包；末步封存后才准入独立CPU参考分析。历史development已反复暴露，不能称研究全程未接触标签或新盲测；Source/DINO预训练交集未知。', 'Phase309已解释1152上下文窗至1022输出的尺寸链，不作为错位证据；最终逐景仿射和高反射冰雪参考语义仍有不确定性。场景snow/ice仅描述属性。本轮为通用预测保持机制对照，非预设云特有创新，不部署、不改阈值/参考、不追加seed或下一phase。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
