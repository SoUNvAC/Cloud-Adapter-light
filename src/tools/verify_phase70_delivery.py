"""Independent local arithmetic and permutation audit; no raster/heldout reads."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def boot(a):
    a=np.asarray(a,float);rng=np.random.default_rng(70010)
    means=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return [float(a.mean()),*np.quantile(means,[.025,.975]),float(np.quantile(means,.95))]


def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)


def main(root):
    d=root/'delivery';r=read(d/'phase70_report.json');lock=read(d/'policy_lock.json');index=read(d/'feature_index.json')
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
    eligible=[]
    for branch in 'ABC':
        gate=r['promotion']['branches'][branch]
        if r['branch_status'][branch]['status']!='completed':assert not gate['eligible'];continue
        p=r['policies'][branch];base=r['policies']['L3'];c=r['comparisons'][branch+'_minus_L3'];delta=np.array([x['delta_ap'] for x in c['pairs']])
        loo=(delta.sum()-delta)/6
        conditions=[delta.mean()>=.01,int((delta>.01).sum())>=4,bool((loo>0).all()),p['macro_recall']>=base['macro_recall'],
            p['fpr_strata']['all']['mean']<=.01,p['miou_percent']>=base['miou_percent'],p['zero_recall_groups']<=base['zero_recall_groups'],
            p['macro_ap']>r['policies'][branch+'_shuffle']['macro_ap']]
        assert conditions==list(gate['conditions'].values()) and all(conditions)==gate['eligible'];close(loo,gate['leave_one_out_ap_means'])
        if all(conditions):eligible.append(branch)
    eligible.sort(key=lambda n:(-r['comparisons'][n+'_minus_L3']['ap']['mean'],6 if n=='A' else 3,n))
    assert eligible[:2]==r['promotion']['selected'] and lock['fit_attempts']<=6
    permutations=0
    for cohort,count in [('fit',153),('development',32)]:
        assert len(index[cohort])==count
        for row in index[cohort]:
            for branch,e in row['branches'].items():
                a=e['permutation'];seed=70010+('ABC'.index(branch)+1)*1000+int(hashlib.sha256(row['product'].encode()).hexdigest()[:8],16)
                assert seed==a['seed'];perm=np.random.default_rng(seed).permutation(a['count'])
                assert hashlib.sha256(perm.tobytes()).hexdigest()==a['sha256'];permutations+=1
    assert len(read(d/'metadata_readiness.json')['rows'])==185
    for key in ['input_lock','feature_index','fit_input_audit','A_reproduction']:
        assert hashlib.sha256((d/(key+'.json')).read_bytes()).hexdigest()==lock[key+'_sha256']
    assert hashlib.sha256((d/'policy_lock.json').read_bytes()).hexdigest()==r['policy_lock_sha256']
    verification=dict(verified=True,csv_rows=len(raw),group_metrics_bootstrap_and_gate_recomputed=True,
        permutation_hashes_recomputed=permutations,file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file()})
    (root/'metric_recomputation.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    paired=[dict(comparison=n,**p) for n,c in r['comparisons'].items() for p in c['pairs']]
    if paired:
        with (root/'paired_group_deltas.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    results(root,r,lock,index)
    print(json.dumps(dict(verified=True,promotion=r['promotion'],metrics={n:{k:p[k] for k in ['macro_ap','macro_recall','miou_percent']} for n,p in r['policies'].items()}),indent=2))


def results(root,r,lock,index,write_summary=True):
    names=['L3','A','A_shuffle','B','B_shuffle','C','C_shuffle'];summary=[];p=r['policies'];fmt=lambda x:f'{x*100:.4f}'
    for n in names:
        if n not in p:summary.append(dict(policy=n,status='failed'));continue
        q=p[n];row=dict(policy=n,status='completed',macro_ap_pct=q['macro_ap']*100,pooled_ap_pct=q['ranking']['ap']*100,
            macro_recall_pct=q['macro_recall']*100,pooled_miou_pct=q['miou_percent'],zero_recall_groups=q['zero_recall_groups'],groups_fpr_above_1pct=q['groups_fpr_above_1pct'])
        for s in ['all','shadow','no_shadow']:
            row[s+'_fpr_pct']=q['fpr_strata'][s]['mean']*100;row[s+'_fpr_upper95_pct']=q['fpr_strata'][s]['upper95']*100
        if n!='L3':
            delta=[(a['ranking']['ap']-b['ranking']['ap'],a['recall']-b['recall']) for a,b in zip(q['rows'],p['L3']['rows']) if a['positive']]
            row['worst_ap_vs_L3_pp']=min(x[0] for x in delta)*100;row['worst_recall_vs_L3_pp']=min(x[1] for x in delta)*100
        summary.append(row)
    fields=list(dict.fromkeys(k for row in summary for k in row))
    if write_summary:
        with (root/'summary.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    lines=['# Phase70 固定三枝探索','','## 1. 数据结果','',
        '原153个fit相关组代表、32个development代表（7组有有效Shadow，25组无Shadow）。原L3在全部32组逐像元分数、AP和confusion精确复现。此次只用已有Source/MsRE缓存；没有新增分割网络前向、下载或确认集评价。',
        f"浅层拟合尝试{lock['fit_attempts']}次，所有完成模型的标准化、参数及全部fit负类校准的全局阈值先冻结，再评价development。原抽样索引及原五列/标签SHA一致。",'',
        'A=两列标准化网络logit×三列标准化光谱，共追加6列；B=61×61排除中心的局部光谱对比，追加3列；C=原生B03/B08/B11的归一化差，追加3列。每枝shuffle均在全image-valid范围共同置换新增列，并单独拟合；不打乱原五列或标签。','',
        '| 策略 | 宏AP% | 池化AP% | 宏实际召回% | 全组FPR%（单侧95%上界） | 池化mIoU% | 零召回组 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        if n not in p:lines.append(f'| {n} | 失败 | — | — | — | — | — |');continue
        q=p[n];s=q['fpr_strata']['all'];lines.append(f"| {n} | {fmt(q['macro_ap'])} | {fmt(q['ranking']['ap'])} | {fmt(q['macro_recall'])} | {fmt(s['mean'])} ({fmt(s['upper95'])}) | {q['miou_percent']:.4f} | {q['zero_recall_groups']} |")
    lines+=['','以下差值单位均为百分点；区间为10000次seed70010配对相关组bootstrap的描述性95%区间。','',
        '| 固定比较 | 宏AP增量 [95%区间] | 宏召回增量 [95%区间] | AP改善/稳定/下降组 |','|---|---:|---:|---:|']
    for n,c in r['comparisons'].items():
        a,b=c['ap'],c['recall'];lines.append(f"| {n} | {fmt(a['mean'])} [{fmt(a['ci95'][0])}, {fmt(a['ci95'][1])}] | {fmt(b['mean'])} [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}] | {c['improved_ap']}/{c['stable_ap']}/{c['declined_ap']} |")
    lines+=['','改善为ΔAP严格>1pp，下降为<-1pp，其他为稳定。无Shadow组AP留空，全部25组仍参与FPR；固定阈值实际召回与逐组CSV中的1%/5%扫描诊断召回分开。',
        '','| 策略 | Shadow FPR%（上界） | 无Shadow FPR%（上界） | 超过1%的组数 |','|---|---:|---:|---:|']
    for n,q in p.items():
        a,b=q['fpr_strata']['shadow'],q['fpr_strata']['no_shadow'];lines.append(f"| {n} | {fmt(a['mean'])} ({fmt(a['upper95'])}) | {fmt(b['mean'])} ({fmt(b['upper95'])}) | {q['groups_fpr_above_1pct']} |")
    lines+=['','### 最差逐组损伤及冻结阈值','', '| 策略 | 拟合迭代数 | fit组平均FPR% | 阈值 | 最差AP相对L3 | 最差召回相对L3 |','|---|---:|---:|---:|---|---|']
    for n,q in p.items():
        rr=[(a,b) for a,b in zip(q['rows'],p['L3']['rows']) if a['positive']]
        worst=min(rr,key=lambda ab:ab[0]['ranking']['ap']-ab[1]['ranking']['ap']);wr=min(rr,key=lambda ab:ab[0]['recall']-ab[1]['recall'])
        model=lock['models'][n]
        lines.append(f"| {n} | {model['iterations']} | {fmt(model['fit_macro_fpr'])} | {model['threshold']:.10f} | {worst[0]['product'].split('_')[-2]} {fmt(worst[0]['ranking']['ap']-worst[1]['ranking']['ap'])}pp | {wr[0]['product'].split('_')[-2]} {fmt(wr[0]['recall']-wr[1]['recall'])}pp |")
    lines+=['','### C分母保护与范围','']
    cr=[e['branches']['C'] for cohort in ['fit','development'] for e in index[cohort] if 'C' in e['branches']]
    if cr:
        count=np.sum([e['protected_count'] for e in cr],axis=0);total=sum(e['observable_pixels'] for e in cr)
        lines.append(f'完成的{len(cr)}产品image-valid像元的三列分母保护计数为{count.tolist()}，分母为{total}，比例为{(count/total).tolist()}。')
        lines.append(f"三列范围：最小{np.min([e['feature_min'] for e in cr],axis=0).tolist()}，最大{np.max([e['feature_max'] for e in cr],axis=0).tolist()}；未截断。逐产品范围、触发比例和置换SHA见feature_index.json。")
    lines+=['','### D几何就绪性','',
        '185产品仅审查已有元数据。逐产品的产品对应关系、标注sidecar CRS与边界有证据；NPY cube的太阳方位/天顶角、CRS/仿射格网、像元大小及栅格朝向缺少可靠对应证据，标记未就绪。没有下载、投影或几何拟合。详见delivery/metadata_readiness.csv及JSON。',
        '','## 2. 结论探讨','',f"冻结晋级规则选中：{r['promotion']['selected'] or '无'}。仅允许提议下一阶段，不能自动启动确认或65c。"]
    for branch,g in r['promotion']['branches'].items():
        if 'conditions' not in g:lines.append(f'{branch}失败：{g["status"]}。');continue
        failed=[k for k,v in g['conditions'].items() if not v]
        lines.append(f"{branch}：{sum(g['conditions'].values())}/8项满足；>1pp改善{g['groups_gt_1pp']}/7组，>0改善{g['groups_gt_zero']}/7组。未满足：{', '.join(failed) or '无'}。去掉任一Shadow组后的AP增量见报告原值。")
    if not r['promotion']['selected']:lines.append('按协议收口，不改阈值、窗口、正则或公式，不追加拟合。')
    if all(n in p for n in ['A','B','B_shuffle','C']):
        lines+=['',
            'A的宏AP、实际召回及池化mIoU均低于L3，本轮固定交互式线性探针没有提供恢复收益；不能据此推断所有非线性机制无效。',
            'B有最值得记录的排序信号：宏AP和池化AP均高于L3，也高于同维数、同拟合规则的打乱对照；去掉任一Shadow组后的平均AP增量仍为正。但4组改善同时伴随3组下降，且最差损伤出现在T32TML。冻结阈值下的宏召回低于L3，排序收益尚未转化为符合本轮约束的整体恢复。匹配对照支持继续把局部光谱对应关系作为一个候选解释，不构成机制证明。',
            'B仅因宏实际召回未达冻结规则而不晋级；AP区间跨零是额外的不确定性说明，不是事后新增的晋级门槛。B的全组FPR点估计下降，但无Shadow组FPR反而上升，且全组单侧95%上界仍超过1%，不能只凭总体均值宣称稳定控制误报。',
            'C的宏AP和宏召回点估计提高，但AP增量未达到1pp，>1pp改善仅2/7组；池化AP还略低于L3。故不能把这一结果称为普遍恢复或改用另一个辅助指标让它通过。']
    lines+=['','这只是7个有Shadow开发组上的多分枝探索，区间有小样本及选择局限，不构成独立确认或普遍机制结论。全部组平均FPR点估计满足1%不等于每景满足；单侧上界超过1%时，稳定预算仍不确定。几何缺输入不等于几何机制无效。原H1失败、Phase66结论及Phase68/69停止状态不变。','',
        '完整精确率、池化召回/FPR、三类IoU、漏影流向、每组分母与扫描诊断在delivery/phase70_report.json及development_groups.csv；所有七行策略及失败状态在summary.csv，已完成策略的配对原值在paired_group_deltas.csv。']
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
