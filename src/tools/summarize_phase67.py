"""Complete development-only reporting for the three necessary controls."""
import argparse,csv,json
from pathlib import Path


def n(x,scale=1):return '未定义' if x is None else f'{x*scale:.4f}'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    d=json.loads((a.root/'delivery/phase67_report.json').read_text())
    lock=json.loads((a.root/'delivery/control_lock.json').read_text())
    p=d['policies'];baselines=d['baseline_reference']
    old_names={'source_original':'Source','msre_original':'原MsRE','msre_calibration':'L1','msre_source':'L2'}
    lines=['# Phase67：三个必要输入对照','', '## 1. 数据结果','',
        '153 fit相关组代表，复用原8192/组无放回样本及组等权拟合；1253376样本、25473个Shadow样本。32 development相关组，仅7组有Shadow、25组无Shadow；未新增分割网络前向，仅三个浅层逻辑回归拟合。',
        '原L3的development像元分数、逐组AP及confusion逐项完全复现。以下原Source/MsRE/L1/L2是已有冻结结果，未重新拟合。', '',
        '| 策略 | 输入维数 | 逐组平均AP(%) | 池化AP(%) | 逐组实际召回(%) | 逐组实际FPR(%) | 池化mIoU(%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in old_names.items():
        q=baselines[key];lines.append(f"| {label} | 原模型或冻结基线 | {n(q['macro_ap'],100)} | {n(q['ranking']['ap'],100)} | {n(q['macro_recall'],100)} | {n(q['macro_fpr'],100)} | {n(q['miou_percent'])} |")
    for name,q in p.items():
        lines.append(f"| {name} | {q['n_features']} | {n(q['macro_ap'],100)} | {n(q['ranking']['ap'],100)} | {n(q['macro_recall'],100)} | {n(q['macro_fpr'],100)} | {n(q['miou_percent'])} |")
    lines += ['', '| 比较 | L3减对照平均AP差(百分点) | 2000次组bootstrap描述95%区间 | 改善/稳定/下降(±1pp) |',
        '|---|---:|---|---|']
    for key,c in d['comparisons'].items():
        lo,hi=c['ap']['descriptive_95_percentile']
        lines.append(f"| {key} | {n(c['ap']['mean'],100)} | [{n(lo,100)}, {n(hi,100)}] | {c['improved']}/{c['stable']}/{c['declined']} |")
    lines+=['', '这些区间只描述7组development，不是三个独立确认检验；没有像元iid显著性。', '',
        '| 策略 | fit冻结阈值(logit) | fit平均FPR(%) | development FPR单侧95%上界(%) | FPR>1%组数 | 无Shadow组平均FPR(%) |',
        '|---|---:|---:|---:|---:|---:|']
    oldlock=json.loads((a.root/'delivery/control_lock.json').read_text())
    for name,q in p.items():
        if name=='L3':threshold=-2.2208902835845947;fit_fpr=.009999992832049201
        else:threshold=oldlock['models'][name]['threshold'];fit_fpr=oldlock['models'][name]['fit_macro_fpr']
        f=q['fpr_constraint_descriptive']
        lines.append(f"| {name} | {n(threshold)} | {n(fit_fpr,100)} | {n(f['upper95_one_sided'],100)} | {f['groups_above_budget']}/32 | {n(q['no_shadow_macro_fpr'],100)} |")
    lines+=['', '| 策略 | 逐组精确率(%)及有效组数 | 池化召回/精确率/FPR(%) | Surface/Cloud/Shadow IoU(%) | Shadow→Surface/Cloud像元 | 池化oracle R@1%/5%FPR(%) |',
        '|---|---|---|---|---|---|']
    for name,q in p.items():
        oracle=q['ranking']['matched_fpr']
        lines.append(f"| {name} | {n(q['macro_precision'],100)} ({q['precision_defined_groups']}/32) | {' / '.join(n(q[k],100) for k in ['recall','precision','fpr'])} | {' / '.join(n(v) for v in q['iou_percent'])} | {q['shadow_to_surface']} / {q['shadow_to_cloud']} | {n(oracle['0.01']['recall'],100)} / {n(oracle['0.05']['recall'],100)} |")
    lines+=['', '精确率仅在有预测Shadow的组定义，无预测不填0；无Shadow但有误报的precision=0。扫描召回是有真值条件的排序诊断，扫描阈值不作为部署阈值。', '',
        '| 原产品代表 | L3−L2_RGB ΔAP(pp) | L3−Source_spectral ΔAP(pp) | L3−MsRE_spectral ΔAP(pp) |',
        '|---|---:|---:|---:|']
    comparisons=list(d['comparisons'].values())
    for i,r in enumerate(comparisons[0]['pairs']):
        assert all(c['pairs'][i]['group_id']==r['group_id'] for c in comparisons)
        lines.append('| '+r['product']+' | '+' | '.join(n(c['pairs'][i]['delta_ap'],100) for c in comparisons)+' |')
    for key,c in d['comparisons'].items():
        w=c['worst_ap'];r=c['worst_recall']
        lines+=['',f"{key}最差AP差：{w['product']} {n(w['delta_ap'],100)}pp；最小实际召回差：{r['product']} {n(r['delta_recall'],100)}pp。"]
    choice=d['selected_on_development']
    lines+=['',f"预设development规则选择：**{choice if choice else '无候选满足预算，不选择'}**。先要求平均实际FPR点估计<=1%，再比较逐组AP，不改阈值。该点估计规则不等同单侧上界也满足预算。",'', '## 2. 结论探讨','']
    for key,c in d['comparisons'].items():
        label=key.removeprefix('L3_minus_');delta=c['ap']['mean']*100
        if label=='L2_RGB':interpret='匹配五输入的低层RGB对照，检视原收益是否仅来自维数与亮度信息'
        elif label=='Source_spectral':interpret='移除MsRE分数，检视Source加光谱是否已足够'
        else:interpret='移除Source分数，检视其是否提供额外恢复能力'
        lines.append(f"- {interpret}：L3在7组平均AP上相对该对照为{delta:+.4f}pp。原始差值、区间和池化操作结果须合看；均值方向不是必要性的证明。")
    rgb=d['comparisons']['L3_minus_L2_RGB'];source=d['comparisons']['L3_minus_Source_spectral'];msre=d['comparisons']['L3_minus_MsRE_spectral']
    if rgb['ap']['mean']>0:
        lines+=['', '匹配五输入后，NIR/SWIR在这7组的平均AP仍高于新增RGB，结果不支持“仅增加输入维数或RGB即可完全复现收益”。但这是多轮查看的development，不能把描述区间当新的独立机制确认；池化5%FPR诊断也应保留。']
    if p['Source_spectral']['macro_ap']>=p['L3']['macro_ap'] and p['Source_spectral']['macro_recall']<p['L3']['macro_recall']:
        lines+=['纯Source+光谱虽然逐组AP不低，却没有转化为统一fit阈值迁移后的实际召回和池化AP；不能由高逐组AP宣布它已足够替代MsRE。三类mIoU的非Shadow分类规则差异不解释AP/FPR/Shadow召回差异。']
    lo,hi=msre['ap']['descriptive_95_percentile']
    if abs(msre['ap']['mean'])<.01 and lo<=0<=hi:
        lines+=['MsRE+光谱与L3平均AP非常接近，Source分数的额外平均收益未在此development中显示清楚。该简化候选是有竞争力的对照；区间跨零不能当作严格等效证明，也不能说Source在每组都无贡献。']
    lines+=['预设选择仍为L3，但不能把极小点估计领先夸大为复杂组合的必要性；不因结果把选择规则临时改为更偏好简化候选。development的FPR单侧上界仍超过1%，点估计满足不代表稳定预算获支持。']
    lines+=['', 'Source_spectral的Surface/Cloud胜者也来自Source，确实完全移除MsRE；另外三个候选采用MsRE。三类mIoU同时包含这个分类规则差异，AP、Shadow召回和FPR不受它影响。',
        '这些是既有7组development上的必要对照和选择，不能宣布新的独立泛化、不能改写Phase66对原冻结L3的结果，也不能把已查看的Phase66确认池用于新候选调试或重复确认。原H1失败及65c未授权保持。',
        '仅完成用户指定三项对照；没有追加波段、交互、正则化或阈值搜索。选择的参数、标准化、fit阈值及输入哈希均保存，任何新独立评价需另立边界。','']
    with (a.root/'RESULTS.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    with (a.root/'group_ap_deltas.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['comparison','group_id','product','delta_ap_pp','delta_recall_pp','delta_fpr_pp']);w.writeheader()
        for name,c in d['comparisons'].items():
            for r in c['pairs']:w.writerow(dict(comparison=name,group_id=r['group_id'],product=r['product'],delta_ap_pp=r['delta_ap']*100,delta_recall_pp=r['delta_recall']*100,delta_fpr_pp=r['delta_fpr']*100))
    model=lock['models'].get(choice) if choice else None
    if choice=='L3':
        model=json.loads((a.root/'reference_recovery_lock.json').read_text())['models']['msre_source_spectral']
    required=[] if choice is None else ['source'] if choice=='Source_spectral' else ['msre'] if choice=='MsRE_spectral' else ['source','msre']
    selected_model=dict(selected_on_development=choice,selection_scope='development only, not independently validated',
        control_lock_sha256=d['control_lock_sha256'],model=model,
        required_checkpoints={name:lock[name+'_checkpoint_sha256'] for name in required},
        original_L3_recovery_lock_sha256=lock['recovery_lock_sha256'],
        source_checkpoint_sha256=lock['source_checkpoint_sha256'],msre_checkpoint_sha256=lock['msre_checkpoint_sha256'])
    with (a.root/'selected_candidate.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(selected_model,f,indent=2)
    print(a.root/'RESULTS.md')


if __name__=='__main__':main()
