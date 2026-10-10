"""Independent local metric audit for four fixed B controls; no raster/heldout reads."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def boot(a):
    a=np.asarray(a,float);rng=np.random.default_rng(71010)
    means=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return [float(a.mean()),*np.quantile(means,[.025,.975]),float(np.quantile(means,.95))]


def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)


def main(root):
    d=root/'delivery';r=read(d/'phase71_report.json');lock=read(d/'policy_lock.json');index=read(d/'feature_index.json')
    initial=read(d/'input_lock.json');audit=read(d/'fit_input_audit.json')
    assert initial['original_split_sha256']=='a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b'
    assert initial['recovery_lock_sha256']=='bcb91cf56d3b796c22bbf60891fb8d045dd8e6af80f5c42a00818fa72501f31f'
    assert audit['original_five_sample_sha256']=='b5ceece265c7b46a8a2a768951aba57aaca7c7ec053b69c0359b6fda66c1e2cf'
    assert audit['sample_y_sha256']=='e659d5dc7a48c4ecd48bf5adad6cdb969b87800751744bb3beb3e00ff38a8a27'
    for fitted in audit['policies'].values():
        assert fitted['sample_count']==1253376 and fitted['sample_positive']==25473
        assert fitted['original_five_sample_sha256']==audit['original_five_sample_sha256'] and fitted['sample_y_sha256']==audit['sample_y_sha256']
        assert len(fitted['rows'])==len({x['group_id'] for x in fitted['rows']})==153
    with (d/'development_groups.csv').open(encoding='utf-8',newline='') as f:raw=list(csv.DictReader(f))
    groups={n:[x for x in raw if x['policy']==n] for n in r['policies']}
    assert len(raw)==32*len(groups)
    for name,rr in groups.items():
        p=r['policies'][name];pos=[x for x in rr if int(x['positive'])>0]
        assert len(rr)==len({x['group_id'] for x in rr})==32 and len(pos)==7
        assert {x['group_id'] for x in rr}=={x['group_id'] for x in groups['L3']}
        close([np.mean([float(x['ap']) for x in pos]),np.mean([float(x['recall']) for x in pos])],[p['macro_ap'],p['macro_recall']])
        close(np.mean([float(x['precision']) for x in rr if x['precision']!='']),p['macro_precision'])
        for stratum,subset in [('all',rr),('shadow',pos),('no_shadow',[x for x in rr if not int(x['positive'])])]:
            s=p['fpr_strata'][stratum];close(boot([float(x['fpr']) for x in subset]),[s['mean'],*s['ci95'],s['upper95']])
        for key in ['positive','negative','true_positive','false_positive','shadow_to_surface','shadow_to_cloud']:
            assert sum(int(x[key]) for x in rr)==p[key]
        close([p['true_positive']/p['positive'],p['false_positive']/p['negative'],p['true_positive']/(p['true_positive']+p['false_positive'])],
            [p['recall'],p['fpr'],p['precision']])
        cm=np.sum([x['confusion'] for x in p['rows']],axis=0);iou=np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm))*100
        close(iou,p['iou_percent']);close(iou.mean(),p['miou_percent'])
        assert sum(int(x['true_positive'])==0 for x in pos)==p['zero_recall_groups']
    for name,c in r['comparisons'].items():
        left,right=name.split('_minus_');paired=[(a,b) for a,b in zip(groups[left],groups[right]) if int(a['positive'])]
        for key in ['ap','recall']:
            s=c[key];close(boot([float(a[key])-float(b[key]) for a,b in paired]),[s['mean'],*s['ci95'],s['upper95']])
        delta=np.array([float(a['ap'])-float(b['ap']) for a,b in paired])
        assert [int((delta>.01).sum()),int((np.abs(delta)<=.01).sum()),int((delta<-.01).sum())]==[c['improved_ap'],c['stable_ap'],c['declined_ap']]
    assert lock['fit_attempts']==4 and set(groups)==set(['L3','B','RGB_local','B_noB08','B_noB11','B_noB12'])
    assert len(raw)==192
    prior=read(root.parents[1]/'phase70/phase70_20261010/delivery/phase70_report.json')
    for name in ['L3','B']:
        for row,old in zip(r['policies'][name]['rows'],prior['policies'][name]['rows']):
            assert row['product']==old['product'] and row['ranking']['ap']==old['ranking']['ap'] and row['confusion']==old['confusion']
    for key in ['input_lock','feature_index','fit_input_audit','baseline_reproduction']:
        assert hashlib.sha256((d/(key+'.json')).read_bytes()).hexdigest()==lock[key+'_sha256']
    assert hashlib.sha256((d/'policy_lock.json').read_bytes()).hexdigest()==r['policy_lock_sha256']
    verification=dict(verified=True,csv_rows=len(raw),group_metrics_bootstrap_recomputed=True,baseline_L3_B_exact=True,
        four_fits_only=True,file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file()})
    (root/'metric_recomputation.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    paired=[dict(comparison=n,**p) for n,c in r['comparisons'].items() for p in c['pairs']]
    table(root/'paired_group_deltas.csv',paired)
    results(root,r,lock)
    print(json.dumps(dict(verified=True,comparisons={n:c['ap'] for n,c in r['comparisons'].items()},metrics={n:{k:p[k] for k in ['macro_ap','macro_recall','miou_percent']} for n,p in r['policies'].items()}),indent=2))


def table(path,rows):
    if path.exists():
        with path.open(encoding='utf-8',newline='') as f:existing=list(csv.DictReader(f))
        expected=[{k:'' if v is None else str(v) for k,v in row.items()} for row in rows]
        assert existing==expected,'Refuse to overwrite a changed result table: '+str(path)
        return
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def results(root,r,lock):
    policies=r['policies'];names=list(policies);pct=lambda v:'—' if v is None else f'{100*v:.3f}'
    summary=[]
    for name,p in policies.items():
        row=dict(policy=name,macro_ap_pct=p['macro_ap']*100,pooled_ap_pct=p['ranking']['ap']*100,macro_recall_pct=p['macro_recall']*100,
            macro_precision_pct=p['macro_precision']*100,pooled_miou_pct=p['miou_percent'],zero_recall_groups=p['zero_recall_groups'])
        for s in ['all','shadow','no_shadow']:
            row[s+'_fpr_pct']=100*p['fpr_strata'][s]['mean'];row[s+'_fpr_upper95_pct']=100*p['fpr_strata'][s]['upper95']
        summary.append(row)
    table(root/'summary.csv',summary)
    maps={n:{x['group_id']:x for x in p['rows']} for n,p in policies.items()};wide=[];damage=[]
    for base in policies['L3']['rows']:
        g=base['group_id'];tile=base['product'].split('_')[-2];b=maps['B'][g]
        row=dict(product=base['product'],tile=tile,group_id=g,positive=base['positive'],negative=base['negative'],prespecified_damage=tile in r['prespecified_damage_tiles'])
        for n in names:
            x=maps[n][g]
            for key,value in [('ap',x['ranking']['ap']),('recall',x['recall']),('fpr',x['fpr'])]:
                row[n+'_'+key+'_pct']=None if value is None else value*100
                for refname,ref in [('L3',base),('B',b)]:
                    rv=ref['ranking']['ap'] if key=='ap' else ref[key]
                    row[n+'_'+key+'_minus_'+refname+'_pp']=None if value is None or rv is None else (value-rv)*100
            if row['prespecified_damage']:
                damage.append(dict(product=x['product'],tile=tile,group_id=g,policy=n,ap_pct=x['ranking']['ap']*100,recall_pct=x['recall']*100,fpr_pct=x['fpr']*100,
                    delta_ap_vs_B_pp=(x['ranking']['ap']-b['ranking']['ap'])*100,delta_recall_vs_B_pp=(x['recall']-b['recall'])*100,
                    delta_ap_vs_L3_pp=(x['ranking']['ap']-base['ranking']['ap'])*100,delta_recall_vs_L3_pp=(x['recall']-base['recall'])*100,
                    true_positive=x['true_positive'],false_positive=x['false_positive'],shadow_to_surface=x['shadow_to_surface'],shadow_to_cloud=x['shadow_to_cloud']))
        wide.append(row)
    table(root/'all_group_changes.csv',wide);table(root/'three_declining_groups.csv',damage)
    lines=['# Phase71：B光谱特异性与逐波段删除','','## 1. 数据结果','',
        '只使用原Catalogue4172871的153个fit代表、32个development代表（7组有有效Shadow）。L3逐像元分数及L3/B全部32组AP/confusion已精确复现，B本轮复现与正式评价的分数SHA一致。只新增四次串行浅层拟合，无新网络前向、数据下载或确认集评价。',
        '原L3五列始终不变。RGB_local追加同61×61、同公式的原生B04/B03/B02局部对比；B_noB08/B_noB11/B_noB12只删除相应新增局部对比列。每版独立fit标准化、原组等权BCE/0.001惩罚和全部fit负类1%组平均FPR校准；全部参数/阈值先冻结。','',
        '| 策略 | 宏AP% | 池化AP% | 宏实际召回% | 全组FPR%（单侧95%上界） | 池化mIoU% |','|---|---:|---:|---:|---:|---:|']
    for n,p in policies.items():
        f=p['fpr_strata']['all'];lines.append(f"| {n} | {pct(p['macro_ap'])} | {pct(p['ranking']['ap'])} | {pct(p['macro_recall'])} | {pct(f['mean'])} ({pct(f['upper95'])}) | {p['miou_percent']:.3f} |")
    lines+=['','### 光谱特异性与删除对照的固定比较','',
        '增量均为百分点，95%区间为10000次seed71010相关组配对bootstrap的描述区间。B−删除版本为正表示该局部波段在当前重新拟合对照中提供正的条件增量。','',
        '| 比较 | 宏AP增量 [95%区间] | 宏召回增量 [95%区间] | AP改善/稳定/下降 |','|---|---:|---:|---:|']
    for n,c in r['comparisons'].items():
        a,b=c['ap'],c['recall'];lines.append(f"| {n} | {pct(a['mean'])} [{pct(a['ci95'][0])}, {pct(a['ci95'][1])}] | {pct(b['mean'])} [{pct(b['ci95'][0])}, {pct(b['ci95'][1])}] | {c['improved_ap']}/{c['stable_ap']}/{c['declined_ap']} |")
    lines+=['','### 全部7个有Shadow组：AP原值（%）','', '| 产品tile | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
    positive=[x for x in policies['L3']['rows'] if x['positive']]
    for x in positive:lines.append('| '+x['product'].split('_')[-2]+' | '+' | '.join(pct(maps[n][x['group_id']]['ranking']['ap']) for n in names)+' |')
    lines+=['','### 全部7个有Shadow组：实际召回原值（%）','', '| 产品tile | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
    for x in positive:lines.append('| '+x['product'].split('_')[-2]+' | '+' | '.join(pct(maps[n][x['group_id']]['recall']) for n in names)+' |')
    lines+=['','### 全部32组：各策略相对原B的FPR变化（百分点）','',
        '无参考Shadow组不计算AP，仍参与完整FPR评价。产品ID、日期、相关组ID和所有绝对值及相对L3/B的AP、召回、FPR差均在all_group_changes.csv。','',
        '| 产品tile/日期 | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
    for x in policies['L3']['rows']:
        b=maps['B'][x['group_id']];parts=x['product'].split('_');lines.append('| '+parts[-2]+'/'+parts[2][:8]+' | '+' | '.join(pct(maps[n][x['group_id']]['fpr']-b['fpr']) for n in names)+' |')
    lines+=['','### 三个预先指定下降组：相对原B的变化（百分点）','',
        '| tile | 策略 | ΔAP | Δ实际召回 | FPR% | Shadow→Surface | Shadow→Cloud |','|---|---|---:|---:|---:|---:|---:|']
    for x in damage:lines.append(f"| {x['tile']} | {x['policy']} | {x['delta_ap_vs_B_pp']:.3f} | {x['delta_recall_vs_B_pp']:.3f} | {x['fpr_pct']:.3f} | {x['shadow_to_surface']} | {x['shadow_to_cloud']} |")
    lines+=['','### 拟合与校准锁','', '| 策略 | 迭代数 | 全局阈值 | fit平均FPR% |','|---|---:|---:|---:|']
    for n,m in lock['models'].items():lines.append(f"| {n} | {m['iterations']} | {m['threshold']:.10f} | {pct(m['fit_macro_fpr'])} |")
    lines+=['','## 2. 结论探讨','', '### 问题一：光谱特异性','']
    c=r['comparisons']['B_minus_RGB_local']['ap']
    lines.append(f"在保留原五输入的条件下，B相对同维数RGB局部对比的宏AP差为{pct(c['mean'])}pp，描述95%区间[{pct(c['ci95'][0])},{pct(c['ci95'][1])}]pp。")
    if c['ci95'][0]>0:lines.append('当前开发组支持新增NIR/SWIR局部对比优于这一固定RGB局部亮度对照；该证据仍是开发探索，不是独立确认。')
    elif c['ci95'][1]<0:lines.append('当前固定RGB对照优于B，不能把B的收益解释为NIR/SWIR局部信息的特异优势。')
    else:lines.append('区间跨零，当前七组证据不能明确区分多光谱局部对比与这一RGB局部亮度对照；不能据点估计宣称特异性，也不能宣称二者等效。')
    lines+=['原五输入本身含B08/B11/B12，因此这不是纯RGB模型与多光谱模型的比较。一个RGB局部对照也不能排除全部亮度、纹理或其他混杂。','', '### 问题二：波段贡献与损伤','']
    for band in ['B08','B11','B12']:
        c=r['comparisons']['B_minus_B_no'+band]['ap'];rec=r['comparisons']['B_minus_B_no'+band]['recall']
        lines.append(f"保留{band}局部对比相对删除它的宏AP条件增量{pct(c['mean'])}pp [{pct(c['ci95'][0])},{pct(c['ci95'][1])}]，宏实际召回条件增量{pct(rec['mean'])}pp。区间和逐组结果需共同判断，不能仅按一个系数或均值给波段定性。")
    eight=r['comparisons']['B_minus_B_noB08']['pairs'];other=[x['delta_ap'] for x in eight if '_T38TNN_' not in x['product']]
    focus=next(x['delta_ap'] for x in eight if '_T38TNN_' in x['product'])
    lines.append(f'B08的平均AP正增量主要集中在T38TNN：该组保留B08局部项比删除高{pct(focus)}pp，而其余六组平均为{pct(np.mean(other))}pp。这是逐组异质性描述，不能据均值宣布B08在多数场景都必要。')
    lines.append('B11局部项的七组AP差均落在±1pp内，当前没有清楚的额外条件收益证据；这不等于B11波段无用，原五输入的原始B11仍保留，且局部光谱列可能存在冗余。B12的AP正增量主要出现在T38TNN和T32NKF，也不是所有组一致获益。三个波段平均删除增量区间均跨零，尚不能独立确认某一个波段普遍必要。')
    for tile in r['prespecified_damage_tiles']:
        rr=[x for x in damage if x['tile']==tile and x['policy'].startswith('B_no')]
        lines.append(tile+'删除对照：'+'；'.join(f"{x['policy']}相对B ΔAP{x['delta_ap_vs_B_pp']:+.3f}pp、Δ召回{x['delta_recall_vs_B_pp']:+.3f}pp" for x in rr)+'。')
    assert all(x['delta_ap_vs_L3_pp']<0 and x['delta_recall_vs_L3_pp']<0 for x in damage if x['policy'].startswith('B_no'))
    lines.append('三个删除版本均未让这三个下降组的AP或实际召回恢复到L3。T35RPM删除B12有小幅改善；T32TML删除任一项仅小幅恢复召回；T40XDR删除后AP略升但召回进一步下降。不存在一个统一的单波段删除方案消除三个组的损伤。')
    lines.append('漏影流向上，T32TML的Shadow→Surface从L3的16140增至B的26049，Shadow→Cloud从636降至527；T40XDR漏影全部流向Surface，45631增至47942；T35RPM两类漏影均增加（Surface 50820→58129，Cloud 11184→15137）。这定位了错误去向，但未证明其地物或物理成因。')
    lines+=['删除版本重新拟合了全部系数、标准化和fit阈值，差值是固定模型族下的条件贡献与损伤关联，不是某波段造成误判的物理因果证明。AP排序恢复与冻结阈值实际召回恢复必须分开看；不能因为删除后某组变好就重新调阈值或组合删除。',
        '仅7个有Shadow开发组、多对照，所有区间均为条件于固定模型的描述性不确定性。原B未晋级及原H1失败保持，本轮不设新的晋级规则、不启动确认或追加搜索。',
        '', '完整逐组精确率、三类IoU、池化指标、漏影流向、分层FPR区间及扫描诊断见delivery/phase71_report.json与development_groups.csv。']
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)


