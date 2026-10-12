"""Post-hoc equal-budget diagnostics; frozen caches, no deployable calibration."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
import run_phase90 as p
from phase91_math import scan, cpu_validate, self_test

ROOT=p.old.SHARED/'phase91_20261012';D=ROOT/'delivery'
P90=p.old.SHARED/'phase90_20261012_run02/delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE91_EQUAL_BUDGET_DIAGNOSTIC_20261012.md'
P90_SHA='1e994f31cf03b593e7fce7962e4b77057910de4072a9aee8635db77262ee2923'
NAMES=('Source_spectral','MsRE_spectral','L3')
BUDGETS=(.01,.005,.02,.05)


def mean(values):
    values=[x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def binary_row(score,truth,threshold,meta):
    positive=truth==2;negative=~positive
    row=dict(**meta,threshold=threshold,positive=int(positive.sum()),negative=int(negative.sum()),
             surface=int(np.count_nonzero(truth==0)),cloud=int(np.count_nonzero(truth==1)))
    if threshold is None:
        row.update(recall=None,fpr=None,precision=None,surface_fpr=None,cloud_fpr=None,
                   true_positive=None,false_positive=None,surface_false_positive=None,cloud_false_positive=None)
        return row
    predicted=score>=threshold;tp=int(np.count_nonzero(predicted&positive));fp=int(np.count_nonzero(predicted&negative))
    sf=int(np.count_nonzero(predicted&(truth==0)));cf=int(np.count_nonzero(predicted&(truth==1)))
    row.update(true_positive=tp,false_positive=fp,surface_false_positive=sf,cloud_false_positive=cf,
        recall=tp/row['positive'] if row['positive'] else None,
        fpr=fp/row['negative'] if row['negative'] else None,
        precision=tp/(tp+fp) if tp+fp else None,
        surface_fpr=sf/row['surface'] if row['surface'] else None,
        cloud_fpr=cf/row['cloud'] if row['cloud'] else None)
    return row


def summary(rows,budget):
    valid=[r for r in rows if r['threshold'] is not None]
    tp=sum(r['true_positive'] for r in valid);fp=sum(r['false_positive'] for r in valid)
    pos=sum(r['positive'] for r in valid);neg=sum(r['negative'] for r in valid)
    surfaces=sum(r['surface'] for r in valid);clouds=sum(r['cloud'] for r in valid)
    recalls=[r['recall'] for r in valid if r['recall'] is not None]
    return dict(rows=rows,macro_recall=mean([r['recall'] for r in valid]),
        macro_fpr=mean([r['fpr'] for r in valid]),macro_precision=mean([r['precision'] for r in valid]),
        macro_surface_fpr=mean([r['surface_fpr'] for r in valid]),macro_cloud_fpr=mean([r['cloud_fpr'] for r in valid]),
        pooled_recall=tp/pos if pos else None,pooled_fpr=fp/neg if neg else None,
        pooled_precision=tp/(tp+fp) if tp+fp else None,
        pooled_surface_fpr=sum(r['surface_false_positive'] for r in valid)/surfaces if surfaces else None,
        pooled_cloud_fpr=sum(r['cloud_false_positive'] for r in valid)/clouds if clouds else None,
        positive=pos,negative=neg,groups=len(rows),recall_groups=len(recalls),
        fpr_groups=sum(r['fpr'] is not None for r in valid),
        no_negative_support_groups=[r['product'] for r in rows if not r['negative']],
        no_shadow_groups=[r['product'] for r in rows if not r['positive']],
        groups_above_budget=sum(r['fpr'] is not None and r['fpr']>budget for r in valid),
        zero_recall_groups=[r['product'] for r in valid if r['recall']==0],
        worst_recall=min(recalls) if recalls else None)


def comparisons(policies,kind,version):
    result={}
    for name in NAMES[:2]:
        pairs=[]
        for a,b in zip(policies['L3']['rows'],policies[name]['rows']):
            assert a['group_id']==b['group_id']
            pairs.append(dict(version=version,kind=kind,control=name,product=a['product'],tile=a['tile'],group_id=a['group_id'],
                delta_recall_pp=(a['recall']-b['recall'])*100 if a['recall'] is not None and b['recall'] is not None else None,
                delta_fpr_pp=(a['fpr']-b['fpr'])*100 if a['fpr'] is not None and b['fpr'] is not None else None))
        eligible=[r for r in pairs if r['delta_recall_pp'] is not None]
        result['L3_minus_'+name]=dict(mean_delta_recall_pp=mean([r['delta_recall_pp'] for r in eligible]),
            pairs=pairs,worst_damage=min(eligible,key=lambda r:r['delta_recall_pp']) if eligible else None,
            primary=(kind=='A' and name=='Source_spectral'),descriptive_only=True,no_postscan_win_test=True)
    return result


def main():
    started=time.monotonic();assert not ROOT.exists(),'Preserve previous attempt'
    p.old.ROOT=p.old.SHARED/'phase89_recovered_20261011'
    assert p.old.digest(P90/'DELIVERY_SHA256.json')==P90_SHA
    for name,sha in p.old.read(P90/'DELIVERY_SHA256.json').items():assert p.old.digest(P90/name)==sha
    lock90=p.old.read(P90/'parameter_input_lock.json');ref=p.old.read(P90/'phase90_report.json')
    assert p.old.digest(p.P67/'control_lock.json')==lock90['phase67_control_sha256']==p.CONTROL_SHA
    assert p.old.digest(p.old.ROOT/'recovery_lock.json')==lock90['phase89_recovery_lock_sha256']
    assert p.old.digest(p.old.ROOT/'delivery/DELIVERY_SHA256.json')==p.P89_MANIFEST_SHA
    p.old.locked()
    for name,sha in lock90['code_sha256'].items():assert p.old.digest(Path(__file__).with_name(name))==sha
    prepared=p.old.read(p.old.ROOT/'prepared/index.json')['rows']
    assert prepared==lock90['prepared']['rows']
    controls={n:p.old.read(p.P67/'control_lock.json')['models'][n] for n in NAMES[:2]}
    models={**controls,'L3':p.old.locked()[2]['msre_source_spectral']}
    ROOT.mkdir();D.mkdir()
    lock=dict(created_utc=p.old.now(),protocol_sha256=p.old.digest(PROTOCOL),phase90_manifest_sha256=P90_SHA,
        phase90_parameter_input_sha256=p.old.digest(P90/'parameter_input_lock.json'),models=models,
        original_thresholds_unchanged=True,readout={n:models[n].get('nonshadow','msre') for n in NAMES},
        source_chain=lock90,score_precision='float64 original60m probabilities; AP diagnostic reproduces float32 tie handling',
        threshold_rule='>=; atomic float64 score ties; lowest observed full score feasible; if none nextafter(max,+inf)',
        macro_constraint='equal mean of FP/Nnegative among groups with Nnegative>0; pooled never substituted',
        cumulative_weights='candidate locator only; refine with per-group integer counts',
        budgets_A=list(BUDGETS),budget_B=.01,oracle_only=True,not_deployment_thresholds=True,
        no_training_fitting_forward_download=True,no_formal_confirmation=True,human_B='not_executed',
        self_tests=self_test(),code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in ['run_phase91.py','phase91_math.py']})
    p.old.write(D/'precision_tie_input_lock.json',lock)
    scores={v:{n:[] for n in NAMES} for v,_,_ in p.VERSIONS};ys={v:[] for v,_,_ in p.VERSIONS}
    metas={v:[] for v,_,_ in p.VERSIONS};reproductions=[];score_seals=[]
    for row in prepared:
        product=row['product'];prob=p.old.probabilities(product)
        new={n:np.empty(row['shape60'],np.float64) for n in NAMES[:2]}
        for y,x,h,w,ss,ms,spec in p.blocks(row):
            for name,m in controls.items():
                xx=p.inputs(ms[...,2],ss[...,2],spec,name)
                new[name][y:y+h,x:x+w]=p.area(p.sigmoid(p.predict_logit(xx,m)).reshape(h*3,w*3),3)
        prob.update(new)
        pred={'L3':p.control_decision(prob['L3'],prob,dict(models['L3'],nonshadow='msre'))}
        for name,m in controls.items():pred[name]=p.control_decision(prob[name],prob,m)
        score_seals.append(dict(product=product,group_id=row['group_id'],
            probability60_raw_sha256={n:p.old.hashlib.sha256(np.ascontiguousarray(prob[n]).tobytes()).hexdigest() for n in NAMES}))
        for version,tkey,mkey in p.VERSIONS:
            truth=p.old.array(row,tkey);mask=p.old.array(row,mkey)
            if not mask.any():continue
            truth=np.asarray(truth[mask]);ys[version].append(truth)
            meta={k:row[k] for k in ['product','group_id','tile']};metas[version].append(meta)
            for name in NAMES:
                score=np.asarray(prob[name][mask],np.float64)
                metric=p.old.row_metrics(truth,pred[name][mask],score)
                reference=next(r for r in ref['versions'][version]['policies'][name]['rows'] if r['product']==product)
                assert metric=={k:reference[k] for k in metric},(version,name,product)
                p.independent_ap(truth,score,metric['ap'])
                cm=[[int(np.count_nonzero((truth==i)&(pred[name][mask]==j))) for j in range(3)] for i in range(3)]
                assert cm==metric['confusion']
                scores[version][name].append(score)
                reproductions.append(dict(version=version,policy=name,**meta,AP_exact=True,
                    frozen_decision_confusion_exact=True,ap=metric['ap'],confusion=metric['confusion']))
        print('Phase91 frozen reproduction',row['tile'],len(reproductions),flush=True)
    assert len(reproductions)==63
    p.old.write(D/'frozen_reproduction.json',dict(rows=reproductions,score_seals=score_seals,no_new_forward=True))
    versions={};Atable=[];Arows=[];Btable=[];Brows=[];frozen_table=[];frozen_groups=[];paired=[];scanlocks=[];cpu=[]
    for version,_,_ in p.VERSIONS:
        policiesA={b:{} for b in BUDGETS};policiesB={};policiesF={}
        for name in NAMES:
            ss=scores[version].pop(name);tt=ys[version];mm=metas[version]
            frozen_rows=[binary_row(s,y,p.sigmoid(models[name]['threshold']),meta) for s,y,meta in zip(ss,tt,mm)]
            for rr in frozen_rows:
                original=next(t for t in ref['versions'][version]['policies'][name]['rows'] if t['product']==rr['product'])
                rr.update(ap=original['ap'],confusion=json.dumps(original['confusion']))
            policiesF[name]=summary(frozen_rows,.01)
            policiesF[name].update(macro_ap=ref['versions'][version]['policies'][name]['macro_ap']['mean'],
                                   pooled_ap=ref['versions'][version]['policies'][name]['pooled']['ap'])
            frozen_table.append(flat(version,name,'frozen',.01,policiesF[name]))
            frozen_groups.extend(dict(version=version,policy=name,**r) for r in frozen_rows)
            points=scan(ss,tt,BUDGETS)
            for point in points:
                cpu_validate(point,ss,tt);budget=point['budget']
                rr=[binary_row(s,y,point['threshold'],meta) for s,y,meta in zip(ss,tt,mm)]
                q=summary(rr,budget);policiesA[budget][name]=dict(q,threshold=point['threshold'],scan=point)
                Atable.append(flat(version,name,'A',budget,q,point['threshold']))
                Arows.extend(dict(version=version,policy=name,kind='A',budget=budget,**r,
                    frozen_to_oracle_recall_pp=(r['recall']-f['recall'])*100 if r['recall'] is not None and f['recall'] is not None else None,
                    frozen_to_oracle_fpr_pp=(r['fpr']-f['fpr'])*100 if r['fpr'] is not None and f['fpr'] is not None else None) for r,f in zip(rr,frozen_rows))
                scanlocks.append(dict(version=version,policy=name,kind='A',point=point))
                cpu.append(dict(version=version,policy=name,kind='A',budget=budget,feasible=True,next_tie_violates=True))
            br=[]
            for s,y,meta in zip(ss,tt,mm):
                point=scan([s],[y],[.01])[0];cpu_validate(point,[s],[y]);br.append(binary_row(s,y,point['threshold'],meta))
                scanlocks.append(dict(version=version,policy=name,kind='B',product=meta['product'],point=point))
                cpu.append(dict(version=version,policy=name,kind='B',product=meta['product'],feasible=True,next_tie_violates=point.get('next_entire_tie_batch_violates')))
            policiesB[name]=summary(br,.01);Btable.append(flat(version,name,'B',.01,policiesB[name]))
            Brows.extend(dict(version=version,policy=name,kind='B',budget=.01,**r) for r in br)
            print('Phase91 scan verified',version,name,flush=True)
        comparisonsA=comparisons(policiesA[.01],'A',version);comparisonsB=comparisons(policiesB,'B',version)
        paired.extend(x for c in [comparisonsA,comparisonsB] for q in c.values() for x in q['pairs'])
        versions[version]=dict(frozen=policiesF,A={str(k):v for k,v in policiesA.items()},B=policiesB,
                               comparisons_A_1percent=comparisonsA,comparisons_B_1percent=comparisonsB)
    report=dict(created_utc=p.old.now(),versions=versions,oracle_only=True,no_new_deployment_threshold=True,
        no_training_fitting_forward_download=True,no_formal_confirmation=True,human_B='not_executed',
        source_score_seals=score_seals,A_vs_B='different constraints; difference is not calibration loss; B need not exceed A',
        no_scan_selected_CI_win_claim=True)
    p.old.write(D/'phase91_report.json',report)
    for filename,rows in [('frozen_workpoints.csv',frozen_table),('frozen_per_group.csv',frozen_groups),
                         ('A_four_budgets.csv',Atable),('A_per_group.csv',Arows),('B_onepercent_table.csv',Btable),
                         ('B_onepercent_per_group.csv',Brows),('paired_recall_changes.csv',paired)]:p.old.csvwrite(D/filename,rows)
    p.old.csvwrite(D/'common_support_tables.csv',[r for r in Atable+Btable+frozen_table if r['version']!='original_main'])
    p.old.write(D/'scan_thresholds_denominators.json',dict(rows=scanlocks,oracle_only=True))
    p.old.write(D/'CPU_VERIFIED.json',dict(self_tests=self_test(),exact_frozen_group_reproduction=63,
        independent_AP_and_confusion=True,scan_points=cpu,checked_points=len(cpu),
        integer_denominators_verified=True,feasible_and_next_tie_checked=True))
    render(report)
    p.old.write(D/'DELIVERY_SHA256.json',{x.name:p.old.digest(x) for x in sorted(D.iterdir()) if x.is_file()})
    p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),elapsed_seconds=time.monotonic()-started,
        delivery_manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json'),new_training=0,new_fit=0,new_forward=0,new_download=0,
        oracle_only=True,no_work_dirs_transfer=True))
    print('Phase91 COMPLETED',flush=True)


def flat(version,name,kind,budget,q,threshold=None):
    return dict(version=version,policy=name,kind=kind,budget=budget,threshold=threshold,
                **{k:v for k,v in q.items() if k not in ['rows'] and not isinstance(v,(list,dict))},
                zero_recall_groups=json.dumps(q['zero_recall_groups']),no_negative_support_groups=json.dumps(q['no_negative_support_groups']))


def render(r):
    def f(x,scale=100):return 'NA' if x is None else f'{x*scale:.4f}'
    lines=['# Phase91：相同误报预算诊断','','## 1．数据结果','',
        '固定Source_spectral、MsRE_spectral、L3；复用Phase90分数链及支持。无训练、拟合、前向或下载。',
        '**全部扫描点是使用当前参考标签得到的事后oracle，不是部署恢复或独立确认。**',
        '','| 原版9组 | 冻结召回 % | 冻结宏FPR % | A全池阈值1%宏预算召回 % | A实际宏FPR % | B逐组1%召回 % |',
        '|---|---:|---:|---:|---:|---:|']
    v=r['versions']['original_main']
    for n in NAMES:
        z=v['frozen'][n];a=v['A']['0.01'][n];b=v['B'][n]
        lines.append('|'+n+'|'+'|'.join([f(z['macro_recall']),f(z['macro_fpr']),f(a['macro_recall']),f(a['macro_fpr']),f(b['macro_recall'])])+'|')
    lines+=['','| 支持范围 | A: L3−Source_spectral召回差 pp | A: L3−MsRE_spectral pp | B: L3−Source_spectral pp | B: L3−MsRE_spectral pp |','|---|---:|---:|---:|---:|']
    for version,v in r['versions'].items():
        values=[v[k]['L3_minus_'+n]['mean_delta_recall_pp'] for k in ['comparisons_A_1percent','comparisons_B_1percent'] for n in NAMES[:2]]
        lines.append('|'+version+'|'+'|'.join(f(x,1) for x in values)+'|')
    lines+=['','| 原版9组A预算 | Source_spectral宏召回 % | MsRE_spectral宏召回 % | L3宏召回 % |','|---|---:|---:|---:|']
    for budget in BUDGETS:lines.append('|'+f(budget)+'%|'+'|'.join(f(r['versions']['original_main']['A'][str(budget)][n]['macro_recall']) for n in NAMES)+'|')
    lines+=['','各组FPR、分母、阈值、超过预算组数、Surface/Cloud分层误报、最差/零召回、池化指标及冻结→oracle变化全部保留CSV/JSON。各参考版本独立扫描自己的oracle阈值，不混称同一工作点。',
        '','## 2．结论分析','']
    deltas=[v['comparisons_A_1percent']['L3_minus_Source_spectral']['mean_delta_recall_pp'] for v in r['versions'].values()]
    if all(x>0 for x in deltas):lines.append('1%宏FPR预算下，L3相对Source＋光谱的平均召回差在三种支持上均正；仅描述该固定预算下的方向，不能扩展为整条PR曲线优势或独立确认。')
    elif all(x<=0 for x in deltas):lines.append('1%宏FPR预算下，L3相对Source＋光谱的平均召回差在三种支持上均非正；降低当前MsRE适配分数必要性的主张。冻结工作点差异不能直接当作排序优势。')
    else:lines.append('1%宏FPR预算下，L3相对Source＋光谱的平均召回差随支持/参考改变方向，未显示稳定优势；降低当前MsRE适配分数必要性的主张。')
    lines+=['A约束相关组平均FPR，B约束每组FPR，两种约束不同；A与B召回之差不是可识别的“校准损失”，B也不保证高于A。',
        '宏AP接近且区间跨零不是等效；无oracle扫描后区间胜出声明。冻结阈值仍原样保留，全部oracle变化未实现为部署收益。',
        '一次诊断不能排除其他机制，不启动新的校准拟合，不选部署赢家、不改标签、成员、人工意见或旧失败结论。',
        'Phase89复用54块缺少故障前输出SHA的历史限制仍在；公开参考不是完整裁决的真值，人工B未执行。',
        '60m对照分数从已校验20m缓存按Phase90原链重算，仅在内存中使用，未保存大数组或同步work_dirs。']
    with (D/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


if __name__=='__main__':main()
