"""Frozen-score post-hoc feasibility, no fits or network inference."""
import importlib.metadata,json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import run_phase94 as r
import run_phase96 as d96
from phase98_math import ap
from phase99_math import oracles,self_test

p=r.p;q=r.q;recovery=r.recovery;fit67=r.fit67
ROOT=p.old.SHARED/'phase99_20261012';D=ROOT/'delivery'
OLD=p.old.SHARED/'phase94_20261012/delivery';P98=p.old.SHARED/'phase98_20261012/delivery'
P96=p.old.SHARED/'phase96_20261012/delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE99_TAIL_CONSTRAINED_ORACLE_20261012.md'
NAMES=('G7_100','SourceS_100','MIN')
def write(n,v):p.old.write(D/n,v)
def sha(a):return q.sha_array(a)
def mean(v):
    a=[x for x in v if x is not None]
    return float(np.mean(a)) if a else None

def score(models,full,checks,meta):
    native={n:r.predict(models,n,r.columns(full,n),checks,meta) for n in r.NAMES}
    assert all(s.dtype==np.float32 for s in native.values())
    g=native['G7_100'].astype(np.float64);z=native['SourceS_100'].astype(np.float64)
    s=np.exp(-np.logaddexp(0.,-z));other=np.empty_like(z);pos=z>=0
    other[pos]=1/(1+np.exp(-z[pos]));e=np.exp(z[~pos]);other[~pos]=e/(1+e)
    assert np.max(np.abs(s-other))<=1e-15
    out=dict(G7_100=g,SourceS_100=s,MIN=np.minimum(g,s))
    assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in out.values())
    assert np.array_equal(out['MIN'],np.where(g<=s,g,s))
    return native,out

def metric(s,y,t,meta,apvalue):
    row=q.binary_row(s,y,t,meta);row['ap']=apvalue
    row['shadow_iou']=row['true_positive']/(row['positive']+row['false_positive']) if row['positive']+row['false_positive'] else None
    # Independent direct counts, separate from binary_row boolean intersections.
    assert row['true_positive']==int(np.sum(s[y==2]>=t))
    assert row['false_positive']==int(np.sum(s[y!=2]>=t))
    return row

def summary(rows,t,pooled_ap,domain,model,kind):
    from run_phase91 import summary as basic
    s=basic(rows,.01);s.update(domain=domain,model=model,kind=kind,threshold=t,
        oracle=kind!='frozen',deployable=False if kind!='frozen' else None,
        macro_ap=mean([a['ap'] for a in rows]),pooled_ap=pooled_ap,
        macro_shadow_iou=mean([a['shadow_iou'] for a in rows]))
    s['max_group_fpr']=max(a['fpr'] for a in rows if a['fpr'] is not None)
    s['tightest_scenes']=[a['product'] for a in rows if a['fpr']==s['max_group_fpr']]
    s['pooled_shadow_iou']=sum(a['true_positive'] for a in rows)/(s['positive']+sum(a['false_positive'] for a in rows))
    return s

def main():
    started=time.monotonic();tests=self_test();assert not ROOT.exists(),'Never overwrite attempt'
    assert p.old.digest(P98/'DELIVERY_SHA256.json')=='63ff2a918cffa8cd2ec5b9c5879371b3fcd6e6114e21a671ccfc18d957ea0756'
    assert p.old.digest(P96/'DELIVERY_SHA256.json')=='791d1abde6a591170bc596ccbeaeec28aa0c017b2d0744dbdf22957f60db1a66'
    prior=p.old.read(P96/'input_lock.json')
    assert p.old.digest(OLD/'DELIVERY_SHA256.json')==prior['phase94_manifest_sha256']
    for root in [OLD,P96,P98]:
        for n,h in p.old.read(root/'DELIVERY_SHA256.json').items():assert p.old.digest(root/n)==h,(root,n)
    sl=p.old.read(OLD/'split_input_lock.json');indices=r.indices_audit()
    assert indices==prior['indices_before']
    assert {n:importlib.metadata.version(n) for n in sl['dependency_lock']['versions']}==sl['dependency_lock']['versions']
    models={}
    for n,m in prior['models'].items():
        assert p.old.digest(OLD/m['file'])==m['sha256']
        if n=='G7_100':
            with (OLD/m['file']).open('rb') as f:models[n]=pickle.load(f)
        else:models[n]=p.old.read(OLD/m['file'])
    oldth=p.old.read(OLD/'four_threshold_lock.json')['models'];minlock=p.old.read(P98/'two_threshold_lock.json')['models']['MIN']
    thresholds=dict(G7_100=float(oldth['G7_100']['cal']['threshold']),SourceS_100=float(r.sigmoid(oldth['SourceS_100']['cal']['threshold'])),MIN=minlock['threshold'])
    audit=p.old.read(fit67.ROOT/'fit_input_audit.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==p.old.read(r.prior.OLD/'fit_input_lock.json')['original_fit_input_audit_sha256']
    am={a['product']:a for a in audit['rows']};recs=recovery.records('fit_representatives');dev=recovery.records('development_val')
    assert len(sl['calibration'])==53 and len(dev)==32
    assert len({a['group_id'] for a in sl['calibration']})==53 and len({a['group_id'] for a in dev.values()})==32
    assert not {a['group_id'] for a in sl['calibration']}&{a['group_id'] for a in dev.values()}
    ROOT.mkdir();D.mkdir()
    code=['run_phase99.py','run_phase99.sh','phase99_math.py','verify_phase99_delivery.py','phase91_math.py','phase98_math.py','phase89_cpu_ap.py','run_phase94.py','run_phase96.py','run_phase91.py','phase93_math.py']
    write('input_lock.json',dict(protocol_sha256=p.old.digest(PROTOCOL),created_utc=p.old.now(),models=prior['models'],
        phase94_manifest_sha256=p.old.digest(OLD/'DELIVERY_SHA256.json'),phase98_manifest_sha256=p.old.digest(P98/'DELIVERY_SHA256.json'),
        phase96_manifest_sha256=p.old.digest(P96/'DELIVERY_SHA256.json'),versions=sl['dependency_lock']['versions'],
        calibration=sl['calibration'],development=list(dev.values()),indices_before=indices,old_frozen_thresholds_probability64=thresholds,
        SourceS_old_logit_threshold=oldth['SourceS_100']['cal']['threshold'],formula='MIN=min(float64(G7_native_float32),stable_sigmoid(float64(SourceS_native_float32_logit)))',
        code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in code},self_tests=tests,
        new_fits=0,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,models_included=list(NAMES),
        constraints=dict(O_mean=dict(mean_fpr=.01),O_tail=dict(mean_fpr=.01,max_group_fpr=.1)),
        oracle_non_deployable=True,no_significance_tests=True,no_bootstrap=True,native_support='original20m full valid',
        AP='noninterpolated, float64 input, independent score-bin and sklearn verification'))
    checks=[];seals=[];policies={};bounds=[];reproductions=[]
    oldcal={n:{a['product']:a for a in oldth[n]['cal']['rows']} for n in r.NAMES}
    oldmincal={a['product']:a for a in minlock['rows']}
    devseals={a['product']:a for a in p.old.read(P98/'input_score_lock.json')['rows']}
    ref98=p.old.read(P98/'phase98_report.json')['policies']
    sources=[{a['product']:a for a in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    for domain,records in [('calibration',[recs[a['product']] for a in sl['calibration']]),('development',list(dev.values()))]:
        scores={n:[] for n in NAMES};ys=[];metas=[]
        for i,rec in enumerate(records):
            product=rec['product'];meta=d96.meta(rec)
            if domain=='calibration':
                recovery.prepare_verify(rec);truth=np.load(recovery.PREP/rec['mask_path']);y=truth[truth!=255]
                full=fit67.full_fit_features(am[product])
            else:
                truth,valid,raw,ss,full=d96.dev_load(rec,sources);y=truth[valid]
                assert sha(y)==devseals[product]['truth_sha256']
            native,out=score(models,full,checks,dict(domain=domain,product=product))
            assert np.array_equal(native['SourceS_100']>=oldth['SourceS_100']['cal']['threshold'],out['SourceS_100']>=thresholds['SourceS_100'])
            for n in r.NAMES:
                if domain=='calibration':assert sha(native[n][y!=2])==oldcal[n][product]['negative_score_sha256']
                else:assert sha(native[n])==devseals[product]['score_sha256'][n]
            assert sha(out['MIN'][y!=2] if domain=='calibration' else out['MIN'])==(oldmincal[product]['negative_score_sha256'] if domain=='calibration' else devseals[product]['score_sha256']['MIN'])
            for n in NAMES:scores[n].append(out[n])
            ys.append(y);metas.append(meta)
            seals.append(dict(domain=domain,**meta,truth_sha256=sha(y),native_score_sha256={n:sha(a) for n,a in native.items()},probability64_sha256={n:sha(a) for n,a in out.items()}))
            del full,native,out
            if domain=='development':del raw,ss
            if (i+1)%10==0 or i+1==len(records):print('Phase99 score recovery',domain,i+1,len(records),flush=True)
        # Reproduce frozen metrics before choosing any oracle in this pool.
        aps={};poolaps={}
        for n in NAMES:
            aps[n]=[ap(s,y) for s,y in zip(scores[n],ys)]
            for s,y,value in zip(scores[n],ys,aps[n]):
                if value is not None:assert abs(float(average_precision_score(y==2,s))-value)<1e-12
            joined=np.concatenate(scores[n]);joined_y=np.concatenate(ys);poolaps[n]=ap(joined,joined_y)
            assert abs(float(average_precision_score(joined_y==2,joined))-poolaps[n])<1e-12
            del joined,joined_y
            rows=[metric(s,y,thresholds[n],dict(**m,domain=domain,model=n,kind='frozen'),a) for s,y,m,a in zip(scores[n],ys,metas,aps[n])]
            if domain=='development':
                refs={a['product']:a for a in ref98[n]['rows']}
                for row in rows:
                    for k in ['ap','recall','fpr','precision','true_positive','false_positive']:assert row[k]==refs[row['product']][k],(n,k)
                assert abs(poolaps[n]-ref98[n]['pooled']['ap'])<1e-12
            else:
                for row in rows:
                    ref=oldmincal[row['product']] if n=='MIN' else oldcal[n][row['product']]
                    assert row['false_positive']==ref['false_positive']
            policies[domain+'/'+n+'/frozen']=summary(rows,thresholds[n],poolaps[n],domain,n,'frozen')
        write(domain+'_frozen_reproduction.json',dict(all_counts_exact=True,AP_exact_development=True if domain=='development' else None,
            SourceS_native_logit_decisions_exact=True,policies={n:policies[domain+'/'+n+'/frozen'] for n in NAMES}))
        reproductions.append(domain)
        for n in NAMES:
            for point in oracles(scores[n],ys):
                t=point['threshold'];kind=point['kind'];assert t is not None
                rows=[metric(s,y,t,dict(**m,domain=domain,model=n,kind=kind),a) for s,y,m,a in zip(scores[n],ys,metas,aps[n])]
                result=summary(rows,t,poolaps[n],domain,n,kind)
                assert result['macro_fpr']==point['macro_fpr'] and result['max_group_fpr']==point['max_group_fpr']
                assert point['false_positives']==[a['false_positive'] for a in rows]
                policies[domain+'/'+n+'/'+kind]=result
                point.update(domain=domain,model=n,groups=metas,unsafe_tie_scenes=[m['product'] for m,c in zip(metas,point['boundary_tie_counts']) if c])
                bounds.append(point)
                print('Phase99 oracle',domain,n,kind,t,result['macro_recall'],result['max_group_fpr'],flush=True)
            a=policies[domain+'/'+n+'/O_mean'];b=policies[domain+'/'+n+'/O_tail']
            assert b['threshold']>=a['threshold'] and b['macro_recall']<=a['macro_recall']
        del scores,ys,aps
    pairs=[];comparisons=[]
    for domain in ['calibration','development']:
        definitions=[('MIN_minus_SourceS_O_tail','MIN/O_tail','SourceS_100/O_tail'),('G7_minus_SourceS_O_tail','G7_100/O_tail','SourceS_100/O_tail')]
        definitions +=[(n+'_tail_minus_mean',n+'/O_tail',n+'/O_mean') for n in NAMES]
        for name,aa,bb in definitions:
            rows=[]
            for a,b in zip(policies[domain+'/'+aa]['rows'],policies[domain+'/'+bb]['rows']):
                assert a['group_id']==b['group_id']
                row=dict(domain=domain,comparison=name,product=a['product'],group_id=a['group_id'],tile=a['tile'],
                    **{'delta_'+k+'_pp':100*(a[k]-b[k]) if a[k] is not None and b[k] is not None else None for k in ['recall','fpr','precision','ap','shadow_iou']})
                rows.append(row)
            pairs+=rows;comparisons.append(dict(domain=domain,comparison=name,mean_delta_recall_pp=mean([a['delta_recall_pp'] for a in rows]),recall_groups=sum(a['delta_recall_pp'] is not None for a in rows)))
    primary=next(a for a in comparisons if a['domain']=='development' and a['comparison']=='MIN_minus_SourceS_O_tail')
    decision='存在值得另立校准研究的经验工作点' if primary['mean_delta_recall_pp']>=5 else '本轮未发现达到既定召回增量的全局阈值工作点'
    assert r.indices_audit()==indices
    write('input_score_lock.json',dict(rows=seals,indices_after=indices))
    write('oracle_boundary_audit.json',dict(points=bounds,float64_nextafter=True,independent_direct_counts=True))
    write('CPU_VERIFIED.json',dict(score_probes=checks,self_tests=tests,frozen_reproduction_domains=reproductions,
        native_score_hashes_exact=True,MIN_phase98_hashes_exact=True,SourceS_old_decisions_exact=True,
        direct_counts=True,AP_sklearn_and_independent_bins=True,AP_threshold_invariant=True,mean_tail_dominance=True))
    report=dict(status='complete',policies=policies,comparisons=comparisons,primary=primary,decision=decision,
        oracle_points=12,frozen_points=6,new_fits=0,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,
        bootstrap_resamples=0,oracle_non_deployable=True,Phase98_branches_remain_closed=True,
        limitations=['Repeatedly exposed development and calibration are not independent confirmation',
            'Oracle uses same-pool reference labels; gain is not achievable deployment calibration gain',
            'Reference semantics not newly human confirmed; human B not performed;54 recovered Phase89 blocks lack pre-crash SHA',
            'Original failed gates remain unchanged'],cost=dict(seconds=time.monotonic()-started,peak_rss_bytes=r.rss(),cpu_threads_max=4))
    write('phase99_report.json',report)
    totals=[{k:v for k,v in a.items() if k!='rows'} for a in policies.values()]
    # Serialize nested audit fields for stable CSV cells.
    totals=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in a.items()} for a in totals]
    p.old.csvwrite(D/'operating_points.csv',totals)
    p.old.csvwrite(D/'per_group.csv',[row for a in policies.values() for row in a['rows']])
    p.old.csvwrite(D/'comparisons.csv',comparisons);p.old.csvwrite(D/'paired_changes.csv',pairs)
    lines=['# Phase99：平均与尾部误报约束下的召回可行性','','## 1．数据结果','',
        '原53组校准池与32组development分别扫描；12个oracle点及6个原冻结点。零拟合／网络前向／下载。所有oracle不可部署。','',
        '|池|评分|工作点|宏召回 %|宏FPR %|最大FPR %|超1%组数|零召回组数|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for a in policies.values():lines.append('|'+ '|'.join([a['domain'],a['model'],a['kind'],*[f'{100*a[k]:.4f}' for k in ['macro_recall','macro_fpr','max_group_fpr']],str(a['groups_above_budget']),str(len(a['zero_recall_groups']))])+'|')
    lines+=['',f"主要比较 development O_tail MIN−SourceS：{primary['mean_delta_recall_pp']:.6f}个百分点，{primary['recall_groups']}个有参考云影组。",'',
        '所有阈值下原分数宏／池化AP完全不变；全表含精确率、Shadow IoU、分层误报、最紧场景和逐组差。无新显著性检验。','',
        '## 2．结论探讨','',decision+'。','Phase98两分枝继续关闭。Oracle阈值使用本池标签，不能用于部署、跨池或声称预算迁移成立。',
        *report['limitations'],'','成本：'+json.dumps(report['cost'])]
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    from verify_phase99_delivery import verify
    acceptance=verify(D);p.old.write(ROOT/'COMPLETED.json',dict(acceptance=acceptance,manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase99 complete',decision,primary,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
