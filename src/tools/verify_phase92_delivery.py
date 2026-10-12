"""Accept sealed Phase92 reports by independent integer counts and group resampling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    for n,h in read('DELIVERY_SHA256.json').items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    if (root/'LOCAL_SUPPLEMENT_SHA256.json').exists():
        for n,h in read('LOCAL_SUPPLEMENT_SHA256.json').items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    lock=read('two_model_lock.json');fi=read('fit_input_lock.json');report=read('phase92_report.json')
    assert hashlib.sha256((root/'fit_input_lock.json').read_bytes()).hexdigest()==lock['fit_input_lock_sha256']
    assert lock['new_logistic_fits']==2 and lock['new_network_forwards']==0
    assert fi['fit_groups']==len(fi['rows'])==153 and fi['positive_features']==0 and not fi['class_balancing']
    assert len({r['group_id'] for r in fi['rows']})==153
    assert sum(r['sample_count'] for r in fi['rows'])==fi['sample_count']
    assert sum(r['sample_positive'] for r in fi['rows'])==fi['sample_positive']
    for n in ['S3','R6']:
        ml=read('model_'+n+'_lock.json');m=ml['model'];assert m==lock['models'][n]
        assert hashlib.sha256((root/('model_'+n+'_lock.json')).read_bytes()).hexdigest()==lock['model_lock_sha256'][n]
        assert ml['fit_input_lock_sha256']==lock['fit_input_lock_sha256']
        assert m['optimizer_success'] and m['positive_features']==0 and m['penalty']==.001
        assert len(m['mean'])==len(m['scale'])==len(m['bands'])==len(m['theta'])-1
        assert all(s>0 for s in m['scale']) and ml['frozen_before_evaluation']
        rr=ml['full_fit_negative_counts'];assert len(rr)==153
        rate=sum(x['false_positive']/x['negative_count'] for x in rr)/153
        assert rate<=.01 and abs(rate-m['fit_macro_fpr'])<1e-15
    count=0
    common=lambda v:[r['group_id'] for r in report['domains'][v]['policies']['S3']['rows']]
    assert common('original_common')==common('revised_common')
    assert set(common('original_common'))<=set(common('original_main'))
    def close(a,b):
        if b is None:assert a is None
        else:assert a is not None and abs(a-b)<1e-12,(a,b)
    def interval(q,values):
        close(q['mean'],float(np.mean(values)) if len(values) else None);assert q['n_groups']==len(values)
        if len(values)>1:
            draws=np.random.default_rng(92012).integers(len(values),size=(10000,len(values)))
            means=np.asarray(values)[draws].mean(1)
            np.testing.assert_allclose(q['ci95'],np.quantile(means,[.025,.975]),rtol=0,atol=1e-12)
            close(q['upper95_one_sided'],float(np.quantile(means,.95)))
    for domain,v in report['domains'].items():
        for kind in ['policies','oracle_1percent']:
            assert set(v[kind])=={'S3','R6','Source_spectral','L3'}
            for n,q in v[kind].items():
                rr=q['rows'];assert len(rr)==dict(development=32,original_main=9,original_common=6,revised_common=6)[domain]
                assert len({r['group_id'] for r in rr})==len(rr)
                for r in [*rr,q['pooled']]:
                    pos=r['positive'];neg=r['negative'];tp=r['true_positive'];fp=r['false_positive']
                    assert fp==r['surface_false_positive']+r['cloud_false_positive'] and neg==r['surface']+r['cloud']
                    close(r['recall'],tp/pos if pos else None);close(r['fpr'],fp/neg if neg else None)
                    close(r['precision'],tp/(tp+fp) if tp+fp else None)
                    close(r['shadow_iou'],tp/(pos+fp) if pos+fp else None)
                    close(r['surface_fpr'],r['surface_false_positive']/r['surface'] if r['surface'] else None)
                    close(r['cloud_fpr'],r['cloud_false_positive']/r['cloud'] if r['cloud'] else None)
                    if not pos:assert r['ap'] is None and r['recall'] is None
                for key in ['true_positive','false_positive','positive','negative','surface','cloud','surface_false_positive','cloud_false_positive']:
                    assert q['pooled'][key]==sum(r[key] for r in rr)
                for metric,it in q['macro'].items():interval(it,[r[metric] for r in rr if r[metric] is not None])
                interval(q['no_shadow_macro_fpr'],[r['fpr'] for r in rr if not r['positive'] and r['fpr'] is not None])
                assert q['groups_above_1percent']==sum(r['fpr'] is not None and r['fpr']>.01 for r in rr)
                assert q['zero_recall_groups']==[r['product'] for r in rr if r['recall']==0]
                if kind=='oracle_1percent':assert q['macro']['fpr']['mean']<=.01
                else:
                    m=lock['models'][n] if n in ['S3','R6'] else read('input_code_score_lock.json')['phase90_chain']['models'][n] if n=='Source_spectral' else None
                    if m is not None:
                        threshold=m['threshold'] if domain=='development' else float(np.exp(-np.logaddexp(0.,-m['threshold'])))
                        close(q['threshold'],threshold)
                count+=len(rr)+1
        for comparison,q in v['comparisons'].items():
            a,b=comparison.split('_minus_');eligible=[]
            for x,y,r in zip(v['policies'][a]['rows'],v['policies'][b]['rows'],q['pairs']):
                assert x['group_id']==y['group_id']==r['group_id']
                for metric in ['ap','recall','fpr']:
                    close(r['delta_'+metric+'_pp'],100*(x[metric]-y[metric]) if x[metric] is not None and y[metric] is not None else None)
                if r['delta_ap_pp'] is not None:eligible.append(r['delta_ap_pp'])
            interval(q['delta_ap_pp'],eligible)
            supported=[r for r in q['pairs'] if r['delta_ap_pp'] is not None]
            recalls=[r for r in q['pairs'] if r['delta_recall_pp'] is not None]
            assert q['worst_ap']==(min(supported,key=lambda r:r['delta_ap_pp']) if supported else None)
            assert q['worst_recall']==(min(recalls,key=lambda r:r['delta_recall_pp']) if recalls else None)
            assert q['improved']==sum(d>1 for d in eligible) and q['stable']==sum(abs(d)<=1 for d in eligible) and q['declined']==sum(d< -1 for d in eligible)
            assert q['primary']==(comparison=='Source_spectral_minus_S3')
    for r in read('oracle_threshold_receipts.json')['rows']:
        point=r['point'];rates=[fp/n for fp,n in zip(point['false_positives'],point['denominators']) if n]
        close(point['macro_fpr'],sum(rates)/len(rates));assert point['macro_fpr']<=.01
        if point['next_batch_false_positives'] is not None:
            rates=[fp/n for fp,n in zip(point['next_batch_false_positives'],point['denominators']) if n]
            assert sum(rates)/len(rates)>.01
    reproductions=read('baseline_reproduction.json')['rows'];assert len(reproductions)==106
    assert all(r['AP_confusion_exact'] for r in reproductions)
    assert len(read('CPU_VERIFIED.json')['block_checks'])==972
    assert report['new_segmentation_training']==report['new_network_forwards']==report['new_downloads']==0
    return dict(sealed_files=len(read('DELIVERY_SHA256.json')),independent_metric_rows=count,baseline_reproductions=106,verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
