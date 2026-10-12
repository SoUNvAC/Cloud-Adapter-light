"""Offline Phase97 SHA, integer metrics, bootstrap and fixed-screen verification."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert digest(n)==h,n
    report=read('phase97_report.json');il=read('input_lock.json');ml=read('model_lock.json');tl=read('threshold_lock.json');fl=read('fit_input_lock.json')
    assert report['status']=='complete' and report['new_fits']==report['new_thresholds']==1
    assert all(report[n]==0 for n in ['new_network_forwards','new_downloads','new_ALCD_pixel_reads'])
    assert digest(ml['file'])==ml['sha256'] and digest('model_lock.json')==tl['model_lock_sha256']
    assert digest('fit_input_lock.json')==ml['fit_input_lock_sha256']
    assert len(il['fit'])==100 and len(il['calibration'])==53 and len(il['development'])==32
    assert len(fl['rows'])==100 and fl['sample_count']==sum(a['count'] for a in fl['rows'])
    assert ml['n_iter']==200 and ml['classes']==[0.,1.] and ml['training_reload_exact']
    A=il['fixed_parameters_A'];B=ml['params'];assert B['monotonic_cst']==[0,0,0,0,0,0,1]
    assert {k:v for k,v in A.items() if k!='monotonic_cst'}=={k:v for k,v in B.items() if k!='monotonic_cst'}
    mono=read('monotonicity_check.json')['models']['B_SourceMonotonic']
    assert mono['point_count']==6400 and mono['max_violation']<=1e-7
    point=tl['exact_boundary']; assert point['macro_fpr']<=.01 and point['next_batch_macro_fpr']>.01
    assert np.isclose(np.mean([a['false_positive']/a['negative_count'] for a in tl['rows']]),point['macro_fpr'],rtol=0,atol=1e-15)
    assert len(tl['rows'])==53 and tl['frozen_before_development']
    def close(a,b):
        if a is None or b is None:assert a is b
        else:assert abs(a-b)<1e-12,(a,b)
    def interval(values):
        x=np.array(values,float);out=dict(n_groups=len(x),mean=float(x.mean()) if len(x) else None,ci95=None,upper95_one_sided=None)
        if len(x)>1:
            means=x[np.random.default_rng(97012).integers(len(x),size=(10000,len(x)))].mean(1)
            out.update(ci95=np.quantile(means,[.025,.975]).tolist(),upper95_one_sided=float(np.quantile(means,.95)))
        return out
    def check_interval(got,values):
        expected=interval(values);assert expected['n_groups']==got['n_groups']
        for k in ['mean','upper95_one_sided']:close(expected[k],got[k])
        if expected['ci95'] is None:assert got['ci95'] is None
        else:np.testing.assert_allclose(expected['ci95'],got['ci95'],rtol=0,atol=1e-12)
    def countrow(a):
        pos=a['positive'];neg=a['negative'];tp=a['true_positive'];fp=a['false_positive']
        assert 0<=tp<=pos and 0<=fp<=neg and a['surface']+a['cloud']==neg
        assert a['surface_false_positive']+a['cloud_false_positive']==fp
        close(a['recall'],tp/pos if pos else None);close(a['fpr'],fp/neg if neg else None)
        close(a['precision'],tp/(tp+fp) if tp+fp else None)
        for k in ['surface','cloud']:close(a[k+'_fpr'],a[k+'_false_positive']/a[k] if a[k] else None)
        if not pos:assert a['ap'] is None
    for name,s in report['policies'].items():
        assert len(s['rows'])==32 and len({a['group_id'] for a in s['rows']})==32
        for a in s['rows']:countrow(a)
        countrow(s['pooled'])
        for k in ['positive','negative','true_positive','false_positive','surface','cloud','surface_false_positive','cloud_false_positive']:assert sum(a[k] for a in s['rows'])==s['pooled'][k]
        for k,t in s['macro'].items():check_interval(t,[a[k] for a in s['rows'] if a[k] is not None])
        close(s['max_group_fpr'],max(a['fpr'] for a in s['rows'] if a['fpr'] is not None))
        assert s['groups_above_1percent']==sum(a['fpr']>.01 for a in s['rows'] if a['fpr'] is not None)
    pairs=report['paired_B_minus_A']['pairs'];assert len(pairs)==32
    for a,b,d in zip(report['policies']['A_G7_100']['rows'],report['policies']['B_SourceMonotonic']['rows'],pairs):
        assert a['group_id']==b['group_id']==d['group_id']
        for k in ['ap','recall','fpr']:close(d['delta_'+k+'_pp'],100*(b[k]-a[k]) if a[k] is not None and b[k] is not None else None)
    for k,t in report['paired_B_minus_A']['intervals_pp'].items():check_interval(t,[a['delta_'+k+'_pp'] for a in pairs if a['delta_'+k+'_pp'] is not None])
    v=report['paired_B_minus_A']['intervals_pp'];b=report['policies']['B_SourceMonotonic']
    criteria=dict(AP_loss_at_most_1pp=v['ap']['mean']>=-1,recall_loss_at_most_1pp=v['recall']['mean']>=-1,
        macro_FPR_at_most_1percent=b['macro']['fpr']['mean']<=.01,maximum_group_FPR_at_most_10percent=b['max_group_fpr']<=.1)
    assert report['screening']==criteria and report['all_four_pass']==all(criteria.values())
    assert len(read('input_score_lock.json')['rows'])==32
    cpu=read('CPU_VERIFIED.json');assert len(cpu['rows'])==53+32*3
    assert all(a['max_error']<=(1e-4 if a.get('model')=='SourceS_100' else 6e-8) for a in cpu['rows'])
    target=read('T31TGF.json');change=target['false_positive_changes']
    assert change['A_only']+change['both']==target['A_G7_100_metrics']['false_positive']
    assert change['B_only']+change['both']==target['B_SourceMonotonic_metrics']['false_positive']
    assert sum(change.values())==target['A_G7_100_metrics']['negative']
    return dict(verified=True,files=len(manifest),fits=1,thresholds=1,development_rows=96,calibration_groups=53,monotonic_points=6400,
        screening=criteria,decision=report['decision'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().root),indent=2))
