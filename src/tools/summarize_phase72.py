"""Local delivery and independent aggregate checks; never access pixel arrays."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'outputs/phase72/phase72_20261010'

def f(v,d=4):return '未定义' if v is None else f'{v:.{d}f}'
def ci(s,scale=1):
    return '不可定义' if s['ci95'] is None else '['+', '.join(f(v*scale) for v in s['ci95'])+']'

def main():
    report=json.loads((ROOT/'delivery/phase72_report.json').read_text(encoding='utf-8'))
    raw=list(csv.DictReader((ROOT/'delivery/groups.csv').open(encoding='utf-8')))
    assert len(raw)==192 and set(r['policy'] for r in raw)=={'B','L3','RGB_local'}
    checks={}
    for name,p in report['policies'].items():
        rr=[r for r in raw if r['policy']==name];assert len(rr)==len({r['group_id'] for r in rr})==64
        cm=np.sum([np.array([[int(r['cm_'+str(i)+str(j)]) for j in range(3)] for i in range(3)]) for r in rr],axis=0)
        assert cm.tolist()==p['confusion']
        for field,target in [('ap','macro_ap'),('recall','macro_recall'),('precision','macro_precision'),('miou_percent','macro_miou_percent')]:
            values=np.array([float(r[field]) for r in rr if r[field]!=''])
            assert np.isclose(values.mean(),p[target]['mean'],rtol=0,atol=1e-12)
            assert len(values)==p[target]['n_groups']
        rng=np.random.default_rng(72010);vals=np.array([float(r['fpr']) for r in rr if r['fpr']!=''])
        boot=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(1)
        assert np.isclose(np.quantile(boot,.95),p['fpr_strata']['all']['upper95_one_sided'],rtol=0,atol=1e-14)
        union=cm.sum(0)+cm.sum(1)-np.diag(cm);iou=np.divide(np.diag(cm),union,out=np.full(3,np.nan),where=union>0)
        assert np.isclose(np.nanmean(iou)*100,p['miou_percent'])
        assert sum(int(r['true_positive'])==0 and int(r['positive'])>0 for r in rr)==len(p['zero_recall_group_ids'])
        checks[name]=dict(all_64_unique=True,pooled_confusion_and_iou_match=True,macro_supports_match=True,fpr_upper_recomputed=True)
    for name,c in report['comparisons'].items():
        b={r['group_id']:r for r in raw if r['policy']=='B'}
        other={r['group_id']:r for r in raw if r['policy']==name}
        for field,key,scale in [('ap','ap_pp',100),('recall','recall_pp',100),('miou_percent','miou_pp',1)]:
            values=np.array([scale*(float(r[field])-float(other[g][field])) for g,r in b.items() if r[field]!='' and other[g][field]!=''])
            rng=np.random.default_rng(72010);boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            assert np.isclose(values.mean(),c[key]['mean'],rtol=0,atol=1e-12)
            assert np.allclose(np.quantile(boot,[.025,.975]),c[key]['ci95'],rtol=0,atol=1e-12)
            assert np.isclose(np.quantile(boot,.05),c[key]['lower95_one_sided'],rtol=0,atol=1e-12)
        checks['B_minus_'+name]=dict(group_means_and_paired_intervals_recomputed=True)
    (ROOT/'verification.json').write_text(json.dumps(dict(status='passed',checks=checks,raw_group_csv_sha256=hashlib.sha256((ROOT/'delivery/groups.csv').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    lines=['# Phase72：完整B独立确认','', '## 1. 数据结果','',
        f"固定65c：64相关组、64原代表；实际有Shadow {report['actual_shadow_support']}组，公开元数据支持{report['public_shadow_support']}组；无重拟合、无阈值修改。",'',
        '| 策略 | 宏AP% | 池化AP% | 宏召回% | 宏精确率% | 宏FPR% | FPR单侧U95% | 宏mIoU% | 池化mIoU% | 零召回组 | FPR>1%组 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in ['B','L3','RGB_local']:
        p=report['policies'][n];fp=p['fpr_strata']['all']
        vals=[n,f(p['macro_ap']['mean']*100),f(p['ranking']['ap']*100),f(p['macro_recall']['mean']*100),f(p['macro_precision']['mean']*100),
            f(fp['mean']*100),f(fp['upper95_one_sided']*100),f(p['macro_miou_percent']['mean']),f(p['miou_percent']),str(len(p['zero_recall_group_ids'])),str(p['groups_fpr_above_1pct'])]
        lines.append('| '+' | '.join(vals)+' |')
    primary=report['comparisons']['L3'];second=report['comparisons']['RGB_local'];b=report['policies']['B'];d=report['decision']
    lines+=['','冻结判据（差值单位为百分点）：','',
        '| 判据 | 实测 | 冻结界限 | 结果 |','|---|---|---|---|',
        f"| B−L3宏AP | {f(primary['ap_pp']['mean'])}，双侧95% {ci(primary['ap_pp'])} | 下界>0 | {d['gates']['ap_increment']} |",
        f"| B−L3宏召回 | {f(primary['recall_pp']['mean'])}，单侧L95 {f(primary['recall_pp']['lower95_one_sided'])} | ≥−1 | {d['gates']['recall_noninferiority']} |",
        f"| B−L3宏mIoU | {f(primary['miou_pp']['mean'])}，单侧L95 {f(primary['miou_pp']['lower95_one_sided'])} | ≥−1 | {d['gates']['miou_noninferiority']} |",
        f"| B宏FPR | {f(b['fpr_strata']['all']['mean']*100)}%，单侧U95 {f(b['fpr_strata']['all']['upper95_one_sided']*100)}% | ≤1% | {d['gates']['fpr_budget']} |",'',
        f"B−RGB_local宏AP：{f(second['ap_pp']['mean'])}pp，95% {ci(second['ap_pp'])}；角色={d['specificity_role']}。",
        f"B−L3改善/稳定/下降={primary['improved']}/{primary['stable']}/{primary['declined']}（±1pp）；B−RGB={second['improved']}/{second['stable']}/{second['declined']}。",'',
        '| 策略 | 池化召回% | 池化精确率% | 池化FPR% | 池化Surface IoU% | Cloud IoU% | Shadow IoU% | 漏影→Surface | 漏影→Cloud |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in ['B','L3','RGB_local']:
        p=report['policies'][n]
        lines.append('| '+' | '.join([n,f(p['recall']*100),f(p['precision']*100),f(p['fpr']*100)]+[f(v) for v in p['iou_percent']]+[str(p['shadow_to_surface']),str(p['shadow_to_cloud'])])+' |')
    lines+=['','分层FPR（逐组等权；单位%，双侧95%区间及单侧U95）：','',
        '| 策略 | 分层 | 组数 | 均值% | 双侧95% | 单侧U95% |','|---|---|---:|---:|---|---:|']
    for n in ['B','L3','RGB_local']:
        for k,s in report['policies'][n]['fpr_strata'].items():lines.append(f"| {n} | {k} | {s['n_groups']} | {f(None if s['mean'] is None else s['mean']*100)} | {ci(s,100)} | {f(None if s['upper95_one_sided'] is None else s['upper95_one_sided']*100)} |")
    lines+=['','有Shadow组逐组结果：','', '| 产品 | B AP% | L3 AP% | RGB AP% | B−L3 ΔAP | B−RGB ΔAP | B−L3 Δ召回 |','|---|---:|---:|---:|---:|---:|---:|']
    maps={n:{r['group_id']:r for r in report['policies'][n]['rows']} for n in ['B','L3','RGB_local']}
    for pair in primary['pairs']:
        if pair['delta_ap_pp'] is None:continue
        g=pair['group_id'];r=maps['B'][g];a=maps['L3'][g];s=maps['RGB_local'][g]
        lines.append('| '+' | '.join([r['product']]+[f(x['ranking']['ap']*100) for x in [r,a,s]]+[f(pair['delta_ap_pp']),f((r['ranking']['ap']-s['ranking']['ap'])*100),f(pair['delta_recall_pp'])])+' |')
    lines+=['',f"最差AP组：{primary['worst_ap']['product']}，ΔAP={f(primary['worst_ap']['delta_ap_pp'])}pp。",
        f"最大召回下降组：{primary['worst_recall']['product']}，Δ召回={f(primary['worst_recall']['delta_recall_pp'])}pp。",'',
        '## 2. 结论探讨','',
        ('B相对L3的AP与全部冻结决策约束获得本批联合支持。' if d['joint_B_vs_L3_confirmed'] else 'B相对L3未通过全部冻结联合条件；改善的辅助指标不能替代失败的主要终点或约束。'),
        ('局部光谱相对匹配RGB对照的AP增量也获得独立支持。' if d['spectral_specificity_confirmed'] else '局部光谱特异性未获本次正式确认；若前置联合规则失败，B−RGB区间只能描述。'),
        '区间按相关组配对bootstrap（10000次、seed72010），条件于已冻结模型；有Shadow组数较少，标签可靠性和未记录访问仍属局限。原H1支持门失败及Phase70未晋级结论保持。确认后不调参、不补样。',
        '', '完整192行、三类混淆、逐组分层FPR、无Shadow组、零召回ID及扫描诊断见delivery/groups.csv和phase72_report.json。所有表中无支持项留空/null；扫描阈值不用于实际结果。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    flat=[]
    for name,c in report['comparisons'].items():
        flat += [dict(comparison='B_minus_'+name,**p) for p in c['pairs']]
    with (ROOT/'paired_group_deltas.csv').open('w',encoding='utf-8-sig',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(dict(verified=True,decision=d,positive_groups=report['actual_shadow_support'])))

if __name__=='__main__':main()
