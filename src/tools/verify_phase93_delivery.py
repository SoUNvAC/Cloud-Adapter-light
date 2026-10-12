"""Independent small-report acceptance; no sklearn import/model fit required."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from phase93_math import PARAMS, COLUMNS


def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert digest(n)==h,n
    lock=read('two_model_lock.json');fi=read('fit_input_lock.json');report=read('phase93_report.json')
    assert lock['new_tree_fits']==2 and report['new_tree_fits']==2
    assert fi['fit_groups']==len(fi['rows'])==153
    assert len({r['group_id'] for r in fi['rows']})==153
    assert fi['sample_count']==sum(r['sample_count'] for r in fi['rows'])==1253376
    assert abs(fi['sample_weight_sum']-fi['sample_count'])<1e-6
    assert fi['standardization'] is False and fi['feature_columns']==COLUMNS
    assert digest('fit_input_lock.json')==lock['fit_input_lock_sha256']
    assert digest('dependency_lock.json')==fi['dependency_lock_sha256']
    assert read('dependency_lock.json')['versions']['scikit-learn']=='1.5.2'
    assert read('baseline_reproduction.json')['all_four_policy_statistics_exact']
    for n,m in lock['models'].items():
        assert digest('model_'+n+'_lock.json')==lock['model_lock_sha256'][n]
        assert m==read('model_'+n+'_lock.json') and digest(n+'.pkl')==m['model_sha256']
        assert m['classes']==[0.,1.] and m['rounds']==200 and m['frozen_before_evaluation']
        assert all(m['params'][k]==v for k,v in PARAMS.items())
        assert 0<=m['threshold']<=1 or m['threshold']==float(np.nextafter(np.float32(1),np.float32(np.inf)))
        rows=m['full_fit_negative_counts'];assert len(rows)==153
        rate=sum(r['false_positive']/r['negative_count'] for r in rows)/153
        assert rate<=.01 and abs(rate-m['fit_macro_fpr'])<1e-15
    for r in read('serialization_checks.json')['rows']:
        assert r['all_training_samples_exact'] and r['samples']==1253376
        assert r['independent_tree_walk_max_error']<=6e-8
    def close(a,b):
        if b is None:assert a is None
        else:assert a is not None and abs(a-b)<1e-12,(a,b)
    def interval(it,values):
        close(it['mean'],float(np.mean(values)) if values else None)
        assert it['n_groups']==len(values) and it['seed']==93012
        if len(values)>1:
            x=np.asarray(values);means=x[np.random.default_rng(93012).integers(len(x),size=(10000,len(x)))].mean(1)
            np.testing.assert_allclose(it['ci95'],np.quantile(means,[.025,.975]),rtol=0,atol=1e-12)
            close(it['upper95_one_sided'],float(np.quantile(means,.95)))
    metrics=0
    for domain,v in report['domains'].items():
        for kind in ['policies','oracle_1percent']:
            assert set(v[kind])=={'S3','R6','Source_spectral','L3','G6','G7'}
            for n,s in v[kind].items():
                rows=s['rows'];assert len(rows)==dict(development=32,original_main=9,original_common=6,revised_common=6)[domain]
                assert len({r['group_id'] for r in rows})==len(rows)
                for r in [*rows,s['pooled']]:
                    tp=r['true_positive'];fp=r['false_positive'];pos=r['positive'];neg=r['negative']
                    assert fp==r['surface_false_positive']+r['cloud_false_positive']
                    assert neg==r['surface']+r['cloud']
                    close(r['recall'],tp/pos if pos else None);close(r['fpr'],fp/neg if neg else None)
                    close(r['precision'],tp/(tp+fp) if tp+fp else None)
                    close(r['shadow_iou'],tp/(pos+fp) if pos+fp else None)
                    close(r['surface_fpr'],r['surface_false_positive']/r['surface'] if r['surface'] else None)
                    close(r['cloud_fpr'],r['cloud_false_positive']/r['cloud'] if r['cloud'] else None)
                    assert 'miou' not in r
                    if not pos:assert r['ap'] is None and r['recall'] is None
                for k in ['true_positive','false_positive','positive','negative','surface','cloud','surface_false_positive','cloud_false_positive']:
                    assert s['pooled'][k]==sum(r[k] for r in rows)
                for k,it in s['macro'].items():interval(it,[r[k] for r in rows if r[k] is not None])
                interval(s['no_shadow_macro_fpr'],[r['fpr'] for r in rows if not r['positive'] and r['fpr'] is not None])
                assert s['groups_above_1percent']==sum(r['fpr'] is not None and r['fpr']>.01 for r in rows)
                assert s['zero_recall_groups']==[r['product'] for r in rows if r['recall']==0]
                if kind=='oracle_1percent':assert s['macro']['fpr']['mean']<=.01
                elif n in COLUMNS:assert s['threshold']==lock['models'][n]['threshold']
                metrics+=len(rows)+1
        assert set(v['comparisons'])=={'G7_minus_G6','Source_spectral_minus_G6','G6_minus_R6'}
        for name,c in v['comparisons'].items():
            a,b=name.split('_minus_');supported=[]
            for x,y,r in zip(v['policies'][a]['rows'],v['policies'][b]['rows'],c['pairs']):
                assert x['group_id']==y['group_id']==r['group_id']
                for k in ['ap','recall','fpr']:close(r['delta_'+k+'_pp'],100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None)
                if r['delta_ap_pp'] is not None:supported.append(r)
            values=[r['delta_ap_pp'] for r in supported];interval(c['delta_ap_pp'],values)
            assert c['improved']==sum(x>1 for x in values) and c['stable']==sum(abs(x)<=1 for x in values) and c['declined']==sum(x< -1 for x in values)
            assert c['worst_ap']==(min(supported,key=lambda r:r['delta_ap_pp']) if supported else None)
            assert c['primary']==(name=='G7_minus_G6')
    common=lambda d:[r['group_id'] for r in report['domains'][d]['policies']['G6']['rows']]
    assert common('original_common')==common('revised_common')
    scans=read('oracle_threshold_receipts.json')['rows'];assert len(scans)==24
    for r in scans:
        s=r['point'];rate=np.mean([fp/n for fp,n in zip(s['false_positives'],s['denominators']) if n])
        close(s['macro_fpr'],float(rate));assert rate<=.01
        if s['next_batch_false_positives'] is not None:
            assert np.mean([fp/n for fp,n in zip(s['next_batch_false_positives'],s['denominators']) if n])>.01
    checks=read('CPU_VERIFIED.json')['independent_tree_walk_checks'];assert len(checks)==64+648
    assert all(r['independent_tree_max_error']<=6e-8 for r in checks)
    assert all(r.get('mean9_max_error',0)<=2e-15 for r in checks)
    assert report['new_segmentation_training']==report['new_network_forwards']==report['new_remote_sensing_downloads']==0
    return dict(verified=True,sealed_files=len(manifest),independent_metric_rows=metrics,
                tree_probe_blocks=len(checks),oracle_points=len(scans))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);a=parser.parse_args()
    print(json.dumps(verify(a.root),indent=2))
