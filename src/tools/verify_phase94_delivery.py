"""Independent Phase94 small-delivery CPU acceptance; never fits a model."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from phase93_math import PARAMS

def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert digest(n)==h,n
    sl=read('split_input_lock.json');fi=read('fit_input_lock.json');ml=read('two_model_lock.json');tl=read('four_threshold_lock.json');report=read('phase94_report.json')
    rows=sl['fit']+sl['calibration'];ranked=sorted(rows,key=lambda r:(hashlib.sha256(('PHASE94_SPLIT_V1|'+r['group_id']).encode()).hexdigest(),r['group_id']))
    assert sl['fit']==ranked[:100] and sl['calibration']==ranked[100:] and len(rows)==153
    assert len({r['group_id'] for r in rows})==153 and sl['sealed_before_truth_arrays']
    assert fi['standardization_groups']==[r['group_id'] for r in sl['fit']]
    assert fi['sample_count']==sum(r['sample_count'] for r in sl['fit'])
    assert abs(fi['tree_weights_sum']-fi['sample_count'])<1e-6 and abs(fi['logistic_weight_sum']-1)<1e-12
    assert fi['no_calibration_group_standardization'] and fi['split_lock_sha256']==digest('split_input_lock.json')
    assert tl['two_model_lock_sha256']==digest('two_model_lock.json') and ml['fit_input_lock_sha256']==digest('fit_input_lock.json')
    assert ml['new_fits']==report['new_fits']==2 and ml['models']['G7_100']['params']==PARAMS
    assert ml['models']['SourceS_100']['params']['positive_features']==1 and ml['models']['SourceS_100']['params']['theta'][0]>=0
    for n,m in ml['models'].items():assert digest(m['file'])==m['sha256']
    assert read('input_score_lock.json')['indices_after']==sl['indices_before']
    assert read('input_score_lock.json')['four_threshold_lock_sha256']==digest('four_threshold_lock.json')
    def close(a,b):
        if b is None:assert a is None
        else:assert a is not None and abs(a-b)<1e-12,(a,b)
    def interval(it,values):
        close(it['mean'],float(np.mean(values)) if values else None)
        assert it['n_groups']==len(values) and it['seed']==94012 and it['resamples']==10000
        if len(values)>1:
            x=np.array(values);means=x[np.random.default_rng(94012).integers(len(x),size=(10000,len(x)))].mean(1)
            np.testing.assert_allclose(it['ci95'],np.quantile(means,[.025,.975]),rtol=0,atol=1e-12)
            close(it['upper95_one_sided'],float(np.quantile(means,.95)))
    for n,m in tl['models'].items():
        for subset,t in m.items():
            assert len(t['rows'])==dict(in_=100,cal=53)[subset if subset!='in' else 'in_']
            eligible=[r for r in t['rows'] if r['negative_count']]
            rate=sum(r['false_positive']/r['negative_count'] for r in eligible)/len(eligible)
            close(t['macro_fpr'],rate);assert rate<=.01
            assert t['next_macro_fpr'] is None or t['next_macro_fpr']>.01
            expected=t['threshold'] if n=='G7_100' else float(np.exp(-np.logaddexp(0.,-t['threshold'])))
            close(t['probability_threshold'],expected)
    for r in read('train_calibration_diagnostics.json')['rows']:
        interval(r['macro_fpr'],[x['false_positive']/x['negative_count'] for x in r['rows'] if x['negative_count']])
        if r['data_subset']==r['threshold_source']:assert r['macro_fpr']['mean']<=.01
    count=0
    for domain,v in report['domains'].items():
        assert set(v['policies'])=={n+'_T_'+s for n in ['G7_100','SourceS_100'] for s in ['in','cal']}
        for name,s in v['policies'].items():
            n,subset=name.split('_T_');assert len(s['rows'])==dict(development=32,original_main=9,original_common=6,revised_common=6)[domain]
            assert len({r['group_id'] for r in s['rows']})==len(s['rows'])
            close(s['threshold'],tl['models'][n][subset]['threshold' if domain=='development' else 'probability_threshold'])
            for r in [*s['rows'],s['pooled']]:
                tp=r['true_positive'];fp=r['false_positive'];pos=r['positive'];neg=r['negative']
                assert fp==r['surface_false_positive']+r['cloud_false_positive'] and neg==r['surface']+r['cloud']
                close(r['recall'],tp/pos if pos else None);close(r['fpr'],fp/neg if neg else None)
                close(r['precision'],tp/(tp+fp) if tp+fp else None);close(r['shadow_iou'],tp/(pos+fp) if pos+fp else None)
                close(r['surface_fpr'],r['surface_false_positive']/r['surface'] if r['surface'] else None)
                close(r['cloud_fpr'],r['cloud_false_positive']/r['cloud'] if r['cloud'] else None)
                if not pos:assert r['ap'] is None and r['recall'] is None
                count+=1
            for k in ['true_positive','false_positive','positive','negative','surface','cloud','surface_false_positive','cloud_false_positive']:assert s['pooled'][k]==sum(r[k] for r in s['rows'])
            for k,it in s['macro'].items():interval(it,[r[k] for r in s['rows'] if r[k] is not None])
            interval(s['no_shadow_macro_fpr'],[r['fpr'] for r in s['rows'] if not r['positive'] and r['fpr'] is not None])
            assert s['groups_above_1percent']==sum(r['fpr'] is not None and r['fpr']>.01 for r in s['rows'])
            assert s['zero_recall_groups']==[r['product'] for r in s['rows'] if r['recall']==0]
        for n in ['G7_100','SourceS_100']:
            a=v['policies'][n+'_T_in'];b=v['policies'][n+'_T_cal']
            assert a['macro']['ap']==b['macro']['ap'] and a['pooled']['ap']==b['pooled']['ap']
            assert [r['ap'] for r in a['rows']]==[r['ap'] for r in b['rows']]
        for name,c in v['comparisons'].items():
            a,b=name.split('_minus_')
            for x,y,r in zip(v['policies'][a]['rows'],v['policies'][b]['rows'],c['pairs']):
                assert r['group_id']==x['group_id']==y['group_id']
                for k in ['ap','fpr','recall','precision','shadow_iou']:close(r['delta_'+k+'_pp'],100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None)
            for k in ['ap','fpr','recall','precision','shadow_iou']:interval(c['delta_'+k+'_pp'],[r['delta_'+k+'_pp'] for r in c['pairs'] if r['delta_'+k+'_pp'] is not None])
            recalls=[r for r in c['pairs'] if r['delta_recall_pp'] is not None]
            assert c['worst_recall']==min(recalls,key=lambda r:r['delta_recall_pp'],default=None)
    for n in ['G7_100','SourceS_100']:
        for s in ['in','cal']:assert [r['group_id'] for r in report['domains']['original_common']['policies'][n+'_T_'+s]['rows']]==[r['group_id'] for r in report['domains']['revised_common']['policies'][n+'_T_'+s]['rows']]
    checks=read('CPU_VERIFIED.json')['rows'];assert len(checks)==712
    assert len(read('calibration_cpu_checks.json')['rows'])==306
    assert all(r['max_error']<=(6e-8 if r['model']=='G7_100' else 1e-4) for r in checks)
    assert all(r.get('mean9_error',0)<=2e-15 for r in checks)
    assert report['new_network_forwards']==report['new_segmentation_training']==report['new_downloads']==0
    return dict(verified=True,sealed_files=len(manifest),independent_metric_rows=count,probe_blocks=len(checks),new_fits=2)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);print(json.dumps(verify(a.parse_args().root),indent=2))
