"""Independent small-delivery acceptance: hashes, metrics, bootstrap and pairs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for name,value in manifest.items():assert sha(root/name)==value,name
    lock=read('parameter_input_lock.json');r=read('phase90_report.json')
    assert sha(root/'phase67_control_lock.json')==lock['phase67_control_sha256']
    assert sha(root/'expert_A_final_original.json')==lock['human_original_sha256']
    assert [lock[k] for k in ['new_training','new_fits','new_forwards','new_downloads']]==[0]*4
    assert r['formal_confirmation'] is False and r['human_B']=='not_executed'
    counts=0
    def stats(values,q):
        a=np.array(values,np.float64);assert len(a)==q['n_groups']
        if not len(a):assert q['mean'] is None;return
        assert abs(float(sum(a)/len(a))-q['mean'])<1e-12
        if len(a)>1:
            indices=np.random.default_rng(90012).integers(len(a),size=(10000,len(a)))
            boot=np.sum(a[indices],axis=1)/len(a)
            np.testing.assert_allclose(np.percentile(boot,[2.5,97.5]),q['ci95'],rtol=0,atol=1e-12)
            assert abs(float(np.percentile(boot,95))-q['upper95_one_sided'])<1e-12
    for version,v in r['versions'].items():
        assert len(v['policies'])==7
        for name,q in v['policies'].items():
            assert len(q['rows'])==(9 if version=='original_main' else 6)
            for row in [*q['rows'],q['pooled']]:
                cm=np.array(row['confusion']);assert cm.shape==(3,3)
                tp=cm[2,2];pos=cm[2].sum();neg=cm[:2].sum();fp=cm[:2,2].sum()
                for k,value in [('recall',float(tp/pos) if pos else None),('fpr',float(fp/neg) if neg else None),
                                ('precision',float(tp/(tp+fp)) if tp+fp else None)]:
                    assert row[k]==value,(name,version,k)
                assert row['shadow_to_surface']==cm[2,0] and row['shadow_to_cloud']==cm[2,1]
                union=cm.sum(0)+cm.sum(1)-np.diag(cm)
                values=[float(cm[i,i]/union[i]*100) if union[i] else None for i in range(3)]
                np.testing.assert_allclose([x for x in row['iou_percent'] if x is not None],[x for x in values if x is not None],atol=1e-12,rtol=0)
                assert abs(np.mean([x for x in values if x is not None])-row['miou_percent'])<1e-12
                counts+=1
            for dest,key in [('macro_ap','ap'),('macro_recall','recall'),('macro_precision','precision'),
                             ('macro_fpr','fpr'),('macro_miou_percent','miou_percent'),
                             ('macro_surface_fpr','surface_fpr'),('macro_cloud_fpr','cloud_fpr')]:
                stats([t[key] for t in q['rows'] if t[key] is not None],q[dest])
            assert q['groups_above_1percent']==sum(t['fpr'] is not None and t['fpr']>.01 for t in q['rows'])
            assert q['zero_recall_groups']==[t['product'] for t in q['rows'] if t['recall']==0]
        for name,q in v['comparisons'].items():
            control=name.removeprefix('L3_minus_');a=v['policies']['L3']['rows'];b=v['policies'][control]['rows']
            for x,y,pair in zip(a,b,q['pairs']):
                assert x['group_id']==y['group_id']==pair['group_id']
                for metric,key,factor in [('ap','delta_ap_pp',100),('recall','delta_recall_pp',100),('miou_percent','delta_miou_pp',1)]:
                    expected=(x[metric]-y[metric])*factor if x[metric] is not None and y[metric] is not None else None
                    assert expected==pair[key]
            delta=[t['delta_ap_pp'] for t in q['pairs'] if t['delta_ap_pp'] is not None]
            stats(delta,q)
            assert [sum(x>1 for x in delta),sum(-1<=x<=1 for x in delta),sum(x< -1 for x in delta)]==[q['improved'],q['stable'],q['declined']]
            assert q['worst_AP']==min(q['pairs'],key=lambda t:t['delta_ap_pp'])
            assert q['worst_recall']==min(q['pairs'],key=lambda t:t['delta_recall_pp'])
    with (root/'seven_policy_tables.csv').open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==21
    with (root/'per_group.csv').open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==147
    cpu=read('CPU_VERIFIED.json');assert cpu['pooled_AP_checks']==21 and cpu['per_group_confusion_and_independent_score_bin_AP_checks']==147
    assert len(read('phase89_reproduction.json')['rows'])==84
    assert len(read('phase67_reproduction.json')['rows'])==96
    result=dict(sealed_files=len(manifest),independent_metric_rows=counts,policies=7,versions=3,
                group_rows=147,pooled_rows=21,all_hashes_metrics_bootstrap_comparisons_verified=True)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    verify(parser.parse_args().root)
