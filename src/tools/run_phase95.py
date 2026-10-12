"""Phase95 zero-fit scale diagnostic; only frozen shallow scoring is permitted."""
import hashlib, importlib.metadata, io, json, pickle, time, zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_phase94 as r
from phase95_math import support, same_scores, threshold, self_test
from verify_phase94_delivery import verify as verify94

p=r.p; q=r.q; recovery=r.recovery; fit67=r.fit67
ROOT=p.old.SHARED/'phase95_20261012'; D=ROOT/'delivery'; OLD=p.old.SHARED/'phase94_20261012/delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE95_SCALE_MATCHED_CALIBRATION_20261012.md'
NAMES=r.NAMES; TYPES=('T20_full','T20_same','T60_same'); SEED=95012


def write(n,v): p.old.write(D/n,v)
def sha(x): return q.sha_array(x)
def probability(n,s): return np.asarray(s,np.float64) if n=='G7_100' else r.sigmoid(s)
def meta(rec): return {k:rec[k] for k in ['product','group_id','tile']} if 'tile' in rec else dict(product=rec['product'],group_id=rec['group_id'],tile=rec['product'].split('_')[-2])


def load_inputs():
    assert not ROOT.exists(), 'Never overwrite or repeat Phase95'
    assert p.old.digest(OLD/'DELIVERY_SHA256.json')=='4ca6d22c435faf25a9b35acfb4cda897c5480a1483a86b1ce19418f78a03c75c'
    accepted=verify94(OLD); sl=p.old.read(OLD/'split_input_lock.json'); ml=p.old.read(OLD/'two_model_lock.json')
    assert {n:importlib.metadata.version(n) for n in sl['dependency_lock']['versions']}==sl['dependency_lock']['versions']
    assert sl['indices_before']==r.indices_audit()
    assert p.old.digest(OLD/'G7_100.pkl')=='2e11a30996b04768b03d6668346d8a34ffda4100c93bbeded7d684e59091c6e6'
    models={}
    for n,m in ml['models'].items():
        assert p.old.digest(OLD/m['file'])==m['sha256']
        if n=='G7_100':
            with (OLD/m['file']).open('rb') as f: models[n]=pickle.load(f)
        else: models[n]=p.old.read(OLD/m['file'])
    base=fit67.base_verify(); audit=p.old.read(fit67.ROOT/'fit_input_audit.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==p.old.read(r.prior.OLD/'fit_input_lock.json')['original_fit_input_audit_sha256']
    assert len(sl['calibration'])==53 and len(sl['fit'])==100
    ROOT.mkdir(); D.mkdir()
    files=['run_phase95.py','phase95_math.py','verify_phase95_delivery.py','run_phase95.sh','run_phase94.py','run_phase92.py','phase89_math.py','phase65d_metrics.py']
    write('input_lock.json',dict(created_utc=p.old.now(),protocol_sha256=p.old.digest(PROTOCOL),
        phase94_manifest_sha256=p.old.digest(OLD/'DELIVERY_SHA256.json'),phase94_acceptance=accepted,
        source_files_sha256=p.old.read(OLD/'DELIVERY_SHA256.json'),model_lock=ml,
        split_lock_sha256=p.old.digest(OLD/'split_input_lock.json'),calibration=sl['calibration'],fit=sl['fit'],
        indices_before=r.indices_audit(),base=base,code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in files},
        rule='crop right/bottom2; top-left1020; homogeneous all-valid nonoverlap3x3; no offset search',
        AP_precision='unchanged Phase94 float32 score-bin AP; decisions and T60_same retain float64',
        new_fits=0,new_network_forwards=0,new_downloads=0,planned_new_thresholds=4))
    return models,sl,{a['product']:a for a in audit['rows']}


def calibrate(models,sl,audit,checks,retention):
    started=time.monotonic(); records=recovery.records('fit_representatives')
    negatives={n:{t:[] for t in TYPES[1:]} for n in NAMES}; receipts=[]
    for i,rec in enumerate(sl['calibration']):
        product=rec['product']; rr=records[product]; recovery.prepare_verify(rr)
        truth=np.load(recovery.PREP/rr['mask_path']); valid=truth!=255
        full=fit67.full_fit_features(audit[product]); assert len(full)==valid.sum()
        keep,y,counts=support(truth); m=meta(dict(rr,product=product)); negative=y!=2
        retention.extend(dict(domain='calibration',**m,**c) for c in counts)
        receipt=dict(**m,mask_sha256=p.old.digest(recovery.PREP/rr['mask_path']),
            blocks_sha256=sha(keep),truth60_sha256=sha(y),negative_blocks=int(negative.sum()),
            negative_pixels=9*int(negative.sum()),eligible=bool(negative.any()),models={})
        for n in NAMES:
            s=r.predict(models,n,r.columns(full,n),checks,dict(domain='calibration',product=product))
            fine,mean,ind,err=same_scores(s,truth,keep,lambda a:probability(n,a))
            a=fine.reshape(-1,9)[negative].ravel(); b=mean[negative]
            negatives[n]['T20_same'].append(a); negatives[n]['T60_same'].append(b)
            receipt['models'][n]=dict(full_score_sha256=sha(s),negative20_sha256=sha(a),negative60_sha256=sha(b),mean9_error=err)
            checks[-1].update(mean9_error=err,same_area_9to1=True)
        receipts.append(receipt); print('Phase95 calibration',i+1,53,flush=True)
    thresholds={}; original=p.old.read(OLD/'four_threshold_lock.json')
    for n in NAMES:
        thresholds[n]={'T20_full':dict(original['models'][n]['cal'],reused_phase94=True)}
        for t in TYPES[1:]:
            arrays=negatives[n][t]; v=threshold(arrays,np.float64 if t=='T60_same' else np.float32)
            v['score_domain']='float64 mean9 Shadow probability' if t=='T60_same' else ('float32 probability' if n=='G7_100' else 'float32 logit')
            v['probability_threshold']=float(v['threshold'] if t=='T60_same' or n=='G7_100' else r.sigmoid(v['threshold']))
            rows=[]; j=0
            for a,receipt in zip(arrays,receipts):
                count=int(np.count_nonzero(a>=v['threshold'])); length=len(a)
                if length: assert count==v['false_positive'][j]; j+=1
                rows.append(dict(product=receipt['product'],group_id=receipt['group_id'],negative_count=length,
                    false_positive=count,fpr=count/length if length else None,negative_score_sha256=sha(a)))
            v['rows']=rows; v['groups']=len(rows); thresholds[n][t]=v
            print('Phase95 threshold',n,t,v['threshold'],v['macro_fpr'],flush=True)
    write('calibration_support_lock.json',dict(rows=receipts,full_negative_scores=True,same_area_weights=True))
    write('four_new_threshold_lock.json',dict(created_utc=p.old.now(),models=thresholds,new_thresholds=4,
        frozen_before_new_evaluation=True,input_lock_sha256=p.old.digest(D/'input_lock.json'),
        calibration_support_lock_sha256=p.old.digest(D/'calibration_support_lock.json'),
        no_evaluation_reference_calibration=True,calibration_seconds=time.monotonic()-started))
    return thresholds


def comparisons(policies):
    out={}
    for n in NAMES:
        for a,b in [('T60_same','T20_same'),('T20_same','T20_full')]:
            pairs=[]
            for x,y in zip(policies[n+'_'+a]['rows'],policies[n+'_'+b]['rows']):
                assert x['group_id']==y['group_id']
                pairs.append(dict(product=x['product'],group_id=x['group_id'],tile=x['tile'],
                    **{'delta_'+k+'_pp':100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None for k in ['ap','fpr','recall','precision','shadow_iou']}))
            out[n+'_'+a+'_minus_'+b]=dict(primary=n=='G7_100' and a=='T60_same',pairs=pairs,
                **{'delta_'+k+'_pp':q.interval([x['delta_'+k+'_pp'] for x in pairs if x['delta_'+k+'_pp'] is not None]) for k in ['ap','fpr','recall','precision','shadow_iou']},
                worst_recall=min((x for x in pairs if x['delta_recall_pp'] is not None),key=lambda x:x['delta_recall_pp'],default=None),
                largest_fpr_increase=max((x for x in pairs if x['delta_fpr_pp'] is not None),key=lambda x:x['delta_fpr_pp'],default=None))
    return out


def evaluate(models,thresholds,checks,retention):
    start=time.monotonic(); results={}; inputs=[]; reproduced=[]
    old_report=p.old.read(OLD/'phase94_report.json'); old_inputs=p.old.read(OLD/'input_score_lock.json')['rows']
    def reproduce(domain,n,s,y,m):
        ref=next(a for a in old_inputs if a['domain']==domain and a['product']==m['product'])
        assert sha(y)==ref['truth_sha256'] and sha(s)==ref['score_sha256'][n],(domain,n,m['product'],'score identity')
        th=thresholds[n]['T20_full']['threshold' if domain=='development' else 'probability_threshold']
        current=q.row_metric(s,y,th,dict(**m,policy=n+'_T_cal'))
        expected=next(a for a in old_report['domains'][domain]['policies'][n+'_T_cal']['rows'] if a['product']==m['product'])
        assert current==expected,(domain,n,m['product'],'Phase94 reproduction')
        reproduced.append(dict(domain=domain,model=n,product=m['product'],score_and_truth_sha_exact=True,metric_row_exact=True))
    def finish(domain,scores,ys,metas):
        policies={}
        for n in NAMES:
            reference=None
            for t in TYPES:
                name=n+'_'+t; th=thresholds[n][t]['probability_threshold']
                rows=[q.row_metric(s,y,th,dict(**m,policy=name)) for s,y,m in zip(scores[n],ys,metas)]
                summary=q.summarize(rows,scores[n],ys,th)
                summary['max_group_fpr']=max((a['fpr'] for a in rows if a['fpr'] is not None),default=None)
                if reference is not None:
                    assert summary['macro']['ap']==reference['macro']['ap'] and summary['pooled']['ap']==reference['pooled']['ap']
                    assert [a['ap'] for a in rows]==[a['ap'] for a in reference['rows']]
                reference=summary; policies[name]=summary
                print('Phase95 summary',domain,name,flush=True)
        results[domain]=dict(policies=policies,comparisons=comparisons(policies),same_model_AP_exact_invariant=True,
            fixed_support='synthetic60m homogeneous' if domain=='development_same60' else 'unchanged Phase94 native60m')
    sources=[{a['product']:a for a in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    scores={n:[] for n in NAMES}; ys=[]; metas=[]
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:
        for product,rec in recovery.records('development_val').items():
            recovery.prepare_verify(rec); truth=np.load(recovery.PREP/rec['mask_path']); valid=truth!=255
            b=z.read('subscenes/'+product+'.npy'); assert hashlib.sha256(b).hexdigest()==rec['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            ss=recovery.load(recovery.OLD,sources[0][product],'score'); ms=recovery.load(recovery.OLD,sources[1][product],'score')
            full=p.combined(recovery.features(raw,ms,ss,valid),raw[...,[3,2,1]][valid]); m=meta(dict(rec,product=product))
            keep,y,counts=support(truth); retention.extend(dict(domain='development_same60',**m,**c) for c in counts); seals={}
            for n in NAMES:
                s=r.predict(models,n,r.columns(full,n),checks,dict(domain='development',product=product)); reproduce('development',n,s,truth[valid],m)
                fine,mean,ind,err=same_scores(s,truth,keep,lambda a:probability(n,a))
                for t in TYPES: assert np.array_equal(mean>=thresholds[n][t]['probability_threshold'],ind>=thresholds[n][t]['probability_threshold'])
                checks[-1].update(mean9_error=err,decisions_exact=True); scores[n].append(mean); seals[n]=sha(mean)
            ys.append(y); metas.append(m); inputs.append(dict(domain='development_same60',**m,truth_sha256=sha(y),block_mask_sha256=sha(keep),score_sha256=seals))
            print('Phase95 development',len(ys),32,flush=True)
    finish('development_same60',scores,ys,metas); del scores,ys,full
    p.old.ROOT=p.old.SHARED/'phase89_recovered_20261011'; p.old.locked(); rows=p.old.read(p.old.ROOT/'prepared/index.json')['rows']
    assert rows==p.old.read(r.prior.OLD/'input_code_score_lock.json')['phase90_chain']['prepared']['rows']
    scores={v:{n:[] for n in NAMES} for v,_,_ in p.VERSIONS}; ys={v:[] for v,_,_ in p.VERSIONS}; metas={v:[] for v,_,_ in p.VERSIONS}
    for i,row in enumerate(rows):
        prob={n:np.empty(row['shape60'],np.float64) for n in NAMES}; coverage=np.zeros(row['shape60'],np.uint8)
        for y,x,h,w,ss,ms,spec in p.blocks(row):
            source=r.logit_score(ss[...,2]).astype(np.float32).ravel()
            features={'G7_100':np.column_stack([spec[...,[2,1,0,3,4,5]].reshape(-1,6),source]).astype(np.float64),
                      'SourceS_100':np.asfortranarray(np.column_stack([source,spec[...,[3,4,5]].reshape(-1,3)]),dtype=np.float64)}
            for n in NAMES:
                s=r.predict(models,n,features[n],checks,dict(domain='ALCD20m',product=row['product'],y60=y,x60=x)).reshape(h*3,w*3)
                a=probability(n,s); value=p.area(a,3); ind=sum(a[dy::3,dx::3] for dy in range(3) for dx in range(3))/9
                err=float(np.max(np.abs(value-ind))); assert err<=2e-15
                for t in TYPES: assert np.array_equal(value>=thresholds[n][t]['probability_threshold'],ind>=thresholds[n][t]['probability_threshold'])
                checks[-1].update(mean9_error=err,decisions_exact=True); prob[n][y:y+h,x:x+w]=value
            coverage[y:y+h,x:x+w]+=1
        assert (coverage==1).all()
        for version,tkey,mkey in p.VERSIONS:
            truth=p.old.array(row,tkey); mask=p.old.array(row,mkey)
            if not mask.any(): continue
            yy=truth[mask]; m=meta(row); ys[version].append(yy); metas[version].append(m); seals={}
            for n in NAMES:
                value=prob[n][mask]; reproduce(version,n,value,yy,m); scores[version][n].append(value); seals[n]=sha(value)
            inputs.append(dict(domain=version,**m,truth_sha256=sha(yy),score_sha256=seals))
        print('Phase95 ALCD',i+1,9,flush=True)
    for v,_,_ in p.VERSIONS: finish(v,scores[v],ys[v],metas[v])
    return results,inputs,reproduced,time.monotonic()-start


def render(report):
    totals=[]; groups=[]; changes=[]
    lines=['# Phase95：固定模型的校准尺度诊断','','## 1．数据结果','','零拟合、零网络前向、零下载。四个新阈值先全部冻结，再评价。',
        'AP沿用Phase94 float32分数并列定义；60m决策、阈值搜索、比较均float64。',
        'development为本轮合成60m同质支持，不与Phase94原20m数值作差。','',
        '|评价池|策略|宏召回%|宏FPR%|单侧95%上界%|精确率%|ShadowIoU%|超1%组|最大FPR%|宏AP%|池化AP%|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    f=lambda x:'未定义' if x is None else f'{100*x:.4f}'
    for domain,v in report['domains'].items():
        for name,s in v['policies'].items():
            row=dict(domain=domain,policy=name,groups=s['groups'],shadow_groups=s['shadow_groups'],threshold=s['threshold'],
                groups_above_1percent=s['groups_above_1percent'],max_group_fpr=s['max_group_fpr'],zero_recall_groups=len(s['zero_recall_groups']))
            for k,it in s['macro'].items(): row.update({f'macro_{k}':it['mean'],f'macro_{k}_ci95':json.dumps(it['ci95']),f'macro_{k}_upper95':it['upper95_one_sided'],f'macro_{k}_groups':it['n_groups']})
            row.update({'pooled_'+k:s['pooled'][k] for k in ['ap','fpr','recall','precision','shadow_iou','surface_fpr','cloud_fpr']})
            totals.append(row); groups.extend(dict(domain=domain,**a) for a in s['rows'])
            lines.append('| '+' | '.join([domain,name,*[f(s['macro'][k]['mean']) for k in ['recall','fpr']],f(s['macro']['fpr']['upper95_one_sided']),*[f(s['macro'][k]['mean']) for k in ['precision','shadow_iou']],str(s['groups_above_1percent']),f(s['max_group_fpr']),f(s['macro']['ap']['mean']),f(s['pooled']['ap'])])+' |')
        for name,c in v['comparisons'].items(): changes.extend(dict(domain=domain,comparison=name,primary=c['primary'],**a) for a in c['pairs'])
    lines+=['','## 2．结论探讨','','T60_same−T20_same隔离固定同区域支持内的概率聚合校准影响；T20_same−T20_full另外描述裁边与同质筛选影响。',
        'AP逐组／池化严格不变，工作点改变不是排序恢复。合成同质内部不能代表真实混合60m像元；算子匹配不能消除跨域差异。',
        '组平均预算满足不等于每组满足；同时查看单侧上界、最大单组误报和召回代价。未选择部署赢家，已查看的ALCD不是独立确认。',
        '逐组配对只含共同有定义组；精确率／IoU可能有不同分母，不能直接用两个宏均值相减代替配对结果。',
        'bootstrap10000、seed95012、相关组等权；人工B未完成和54块恢复缓存无故障前SHA限制不变。',
        '所有Phase94 T_cal逐组原始分数SHA、标签SHA与指标完全复现；旧20m结果仅作链路验收，不用于合成60m效果差值。',
        '详细配对区间／最差组见phase95_report.json，支持保留量见support_retention.csv。']
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name,rows in [('six_strategy_totals.csv',totals),('per_group.csv',groups),('paired_changes.csv',changes),('support_retention.csv',report['support_retention'])]: p.old.csvwrite(D/name,rows)


def main():
    start=time.monotonic(); tests=self_test(); models,sl,audit=load_inputs(); checks=[]; retention=[]
    thresholds=calibrate(models,sl,audit,checks,retention)
    sealed=p.old.digest(D/'four_new_threshold_lock.json')
    results,inputs,reproduced,seconds=evaluate(models,thresholds,checks,retention)
    assert sealed==p.old.digest(D/'four_new_threshold_lock.json') and sl['indices_before']==r.indices_audit()
    for n,m in p.old.read(OLD/'two_model_lock.json')['models'].items(): assert p.old.digest(OLD/m['file'])==m['sha256']
    write('input_score_lock.json',dict(rows=inputs,indices_after=r.indices_audit(),four_threshold_lock_sha256=sealed,no_large_arrays_saved=True))
    write('CPU_VERIFIED.json',dict(tests=tests,rows=checks,phase94_reproduction=reproduced,all_phase94_T_cal_rows_exact=True,
        independent_counts_and_score_bin_AP=True,all_same_model_AP_invariants=True,all_mean9_and_decision_checks=True))
    write('cost.json',dict(total_seconds=time.monotonic()-start,evaluation_seconds=seconds,process_peak_rss_bytes=r.rss(),
        peak_scope='single process cumulative input verification, calibration and evaluation',cpu_threads_max=4,
        new_fits=0,new_network_forwards=0,new_downloads=0,new_thresholds=4))
    report=dict(status='complete',created_utc=p.old.now(),domains=results,support_retention=retention,
        bootstrap_seed=SEED,resamples=10000,new_fits=0,new_network_forwards=0,new_downloads=0,new_thresholds=4,
        exploratory_only=True,no_deployment_winner=True,original_failed_gates_unchanged=True)
    write('phase95_report.json',report); render(report)
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    from verify_phase95_delivery import verify
    acceptance=verify(D); p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),acceptance=acceptance,delivery_manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase95 complete',flush=True)


if __name__=='__main__':
    q.SEED=SEED
    with threadpool_limits(limits=4): main()
