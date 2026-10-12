"""Offline Phase98 hashes, fixed boundaries, counts, intervals and screening."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert digest(n)==h,n
    lock=read('input_formula_lock.json');thresholds=read('two_threshold_lock.json');report=read('phase98_report.json');cpu=read('CPU_VERIFIED.json')
    assert report['status']=='complete' and report['new_thresholds']==2
    for n in ['new_fits','new_network_forwards','new_downloads','new_ALCD_pixel_reads']:assert report[n]==lock[n]==0
    assert lock['formulas']==dict(AVG='(pG+pS)/2',MIN='minimum(pG,pS)') and lock['fusion_dtype']=='float64' and not lock['Phase97_model_included']
    assert len(lock['calibration'])==53 and len(lock['development'])==32
    assert not {a['group_id'] for a in lock['calibration']}&{a['group_id'] for a in lock['development']}
    assert thresholds['input_formula_lock_sha256']==digest('input_formula_lock.json') and thresholds['all_frozen_before_development']
    assert set(thresholds['models'])=={'AVG','MIN'}
    close=lambda a,b:bool(a is b) if a is None or b is None else abs(a-b)<1e-12
    for n,t in thresholds['models'].items():
        point=t['exact_boundary'];rows=t['rows'];assert len(rows)==53 and t['score_dtype']=='float64'
        actual=[a['false_positive']/a['negative_count'] for a in rows if a['negative_count']]
        assert close(float(np.mean(actual)),point['macro_fpr']) and point['macro_fpr']<=.01
        assert point['next_batch_macro_fpr'] is None or point['next_batch_macro_fpr']>.01
        assert point['denominators']==[a['negative_count'] for a in rows] and point['false_positives']==[a['false_positive'] for a in rows]
        for a in rows:assert close(a['fpr'],a['false_positive']/a['negative_count'] if a['negative_count'] else None)
    def intervalcheck(got,values):
        x=np.asarray(values,float);assert got['n_groups']==len(x) and close(got['mean'],float(x.mean()) if len(x) else None)
        if len(x)>1:
            means=x[np.random.default_rng(98012).integers(len(x),size=(10000,len(x)))].mean(1)
            np.testing.assert_allclose(got['ci95'],np.quantile(means,[.025,.975]),rtol=0,atol=1e-12)
            assert close(got['upper95_one_sided'],float(np.quantile(means,.95)))
        else:assert got['ci95'] is None and got['upper95_one_sided'] is None
    def counts(a):
        pos=a['positive'];neg=a['negative'];tp=a['true_positive'];fp=a['false_positive']
        assert 0<=tp<=pos and 0<=fp<=neg and a['surface']+a['cloud']==neg
        assert a['surface_false_positive']+a['cloud_false_positive']==fp
        assert close(a['recall'],tp/pos if pos else None) and close(a['fpr'],fp/neg if neg else None)
        assert close(a['precision'],tp/(tp+fp) if tp+fp else None)
        for k in ['surface','cloud']:assert close(a[k+'_fpr'],a[k+'_false_positive']/a[k] if a[k] else None)
        if not pos:assert a['ap'] is None
    policies=report['policies'];assert set(policies)=={'G7_100','SourceS_100','AVG','MIN'}
    for n,s in policies.items():
        assert len(s['rows'])==32 and len({a['group_id'] for a in s['rows']})==32
        if n in ['AVG','MIN']:assert s['threshold']==thresholds['models'][n]['threshold']
        else:assert s['threshold']==lock['baseline_thresholds'][n]['cal']['threshold']
        for a in s['rows']:counts(a)
        counts(s['pooled'])
        for k in ['positive','negative','true_positive','false_positive','surface','cloud','surface_false_positive','cloud_false_positive']:assert sum(a[k] for a in s['rows'])==s['pooled'][k]
        for k,v in s['macro'].items():intervalcheck(v,[a[k] for a in s['rows'] if a[k] is not None])
        assert close(s['max_group_fpr'],max(a['fpr'] for a in s['rows'] if a['fpr'] is not None))
        assert s['groups_above_1percent']==sum(a['fpr']>.01 for a in s['rows'] if a['fpr'] is not None)
    for na,nb in [('AVG','SourceS_100'),('MIN','SourceS_100'),('MIN','AVG')]:
        c=report['comparisons'][na+'_minus_'+nb];assert len(c['pairs'])==32
        for a,b,d in zip(policies[na]['rows'],policies[nb]['rows'],c['pairs']):
            assert a['group_id']==b['group_id']==d['group_id']
            for k in ['ap','recall','fpr']:assert close(d['delta_'+k+'_pp'],100*(a[k]-b[k]) if a[k] is not None and b[k] is not None else None)
        for k,v in c['intervals_pp'].items():intervalcheck(v,[a['delta_'+k+'_pp'] for a in c['pairs'] if a['delta_'+k+'_pp'] is not None])
    retained=[]
    for n in ['AVG','MIN']:
        v=report['comparisons'][n+'_minus_SourceS_100']['intervals_pp'];s=policies[n]
        criteria=dict(AP_loss_at_most_1pp=v['ap']['mean']>=-1,recall_gain_at_least_5pp=v['recall']['mean']>=5,
            macro_FPR_at_most_1percent=s['macro']['fpr']['mean']<=.01,maximum_group_FPR_at_most_10percent=s['max_group_fpr']<=.1)
        assert report['screening'][n]==dict(criteria=criteria,all_four_pass=all(criteria.values()))
        if all(criteria.values()):retained.append(n)
    assert report['retained_branches']==retained
    assert report['decision']==('eligible_for_separate_protocol' if retained else 'close_both_fixed_fusion_branches')
    assert len(cpu['score_probes'])==170 and len(cpu['fusion_checks'])==85
    for a in cpu['score_probes']:assert a['max_error']<=(1e-4 if a['model']=='SourceS_100' else 6e-8)
    for a in cpu['fusion_checks']:assert a['fusion_dtype']=='float64' and a['sigmoid_independent_max_error']<=1e-15 and a['MIN_bounded'] and a['AVG_between'] and a['independent_formula_exact']
    scorelock=read('input_score_lock.json');assert len(scorelock['rows'])==32
    assert scorelock['two_threshold_lock_sha256']==digest('two_threshold_lock.json') and scorelock['indices_after']==lock['indices_before']
    for a in scorelock['rows']:assert a['score_dtypes']==dict(G7_100='float32',SourceS_100='float32',AVG='float64',MIN='float64')
    target=read('T31TGF.json')
    for n in policies:
        counts(target[n+'_metrics']);assert target[n+'_metrics']['ap'] is None and target[n+'_metrics']['recall'] is None
    return dict(verified=True,files=len(manifest),new_fits=0,new_thresholds=2,development_rows=128,paired_rows=96,
        calibration_groups=53,fusion_dtype='float64',score_probes=170,fusion_checks=85,retained_branches=retained,decision=report['decision'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().root),indent=2))
