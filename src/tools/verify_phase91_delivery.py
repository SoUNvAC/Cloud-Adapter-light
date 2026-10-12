"""Independent acceptance of Phase91 small diagnostics, no raster/model access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(root):
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8-sig'))
    manifest=read('DELIVERY_SHA256.json')
    for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    if (root/'LOCAL_SUPPLEMENT_SHA256.json').exists():
        for name,sha in read('LOCAL_SUPPLEMENT_SHA256.json').items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    report=read('phase91_report.json');lock=read('precision_tie_input_lock.json')
    assert report['oracle_only'] and report['no_new_deployment_threshold'] and report['no_formal_confirmation']
    scans=read('scan_thresholds_denominators.json')['rows'];assert len(scans)==99
    for receipt in scans:
        p=receipt['point'];n=np.asarray(p['denominators']);valid=n>0
        if p['threshold'] is None:assert not valid.any();continue
        rate=sum(float(a)/int(b) for a,b in zip(p['false_positives'],n) if b)/int(valid.sum())
        assert rate<=p['budget'] and abs(rate-p['macro_fpr'])<1e-15
        if p['next_lower_observed_threshold'] is not None:
            next_rate=sum(float(a)/int(b) for a,b in zip(p['next_batch_false_positives'],n) if b)/int(valid.sum())
            assert next_rate>p['budget'] and abs(next_rate-p['next_batch_macro_fpr'])<1e-15
            assert p['threshold']>p['next_lower_observed_threshold']
    count=0
    for version,v in report['versions'].items():
        for kind,policies,budget in [('frozen',v['frozen'],.01),('B',v['B'],.01)]+[('A',x,float(b)) for b,x in v['A'].items()]:
            for name,q in policies.items():
                rows=q['rows'];assert len(rows)==(9 if version=='original_main' else 6)
                for r in rows:
                    if r['threshold'] is None:continue
                    tp=r['true_positive'];fp=r['false_positive'];pos=r['positive'];neg=r['negative']
                    assert r['recall']==(tp/pos if pos else None)
                    assert r['fpr']==(fp/neg if neg else None)
                    assert r['precision']==(tp/(tp+fp) if tp+fp else None)
                    assert fp==r['surface_false_positive']+r['cloud_false_positive']
                    assert neg==r['surface']+r['cloud']
                    if kind=='B' and r['fpr'] is not None:assert r['fpr']<=budget
                    if kind=='frozen':
                        expected=float(np.exp(-np.logaddexp(0.,-lock['models'][name]['threshold'])))
                        assert expected==r['threshold']
                for metric in ['recall','fpr','surface_fpr','cloud_fpr','precision']:
                    values=[r[metric] for r in rows if r[metric] is not None]
                    expected=float(sum(values)/len(values)) if values else None
                    actual=q['macro_'+metric]
                    assert expected is None and actual is None or expected is not None and abs(expected-actual)<1e-14
                if kind=='A':assert q['macro_fpr']<=budget
                assert q['groups_above_budget']==sum(r['fpr'] is not None and r['fpr']>budget for r in rows)
                assert q['zero_recall_groups']==[r['product'] for r in rows if r['recall']==0]
                good=[r for r in rows if r['threshold'] is not None]
                tp=sum(r['true_positive'] for r in good);fp=sum(r['false_positive'] for r in good)
                pos=sum(r['positive'] for r in good);neg=sum(r['negative'] for r in good)
                assert q['pooled_recall']==(tp/pos if pos else None)
                assert q['pooled_fpr']==(fp/neg if neg else None)
                count+=len(rows)
        for kind in ['A','B']:
            comparisons=v['comparisons_'+kind+'_1percent'];policies=v['A']['0.01'] if kind=='A' else v['B']
            for name,q in comparisons.items():
                control=name.removeprefix('L3_minus_');delta=[]
                for x,y,rr in zip(policies['L3']['rows'],policies[control]['rows'],q['pairs']):
                    assert x['group_id']==y['group_id']==rr['group_id']
                    if x['recall'] is None or y['recall'] is None:assert rr['delta_recall_pp'] is None;continue
                    d=(x['recall']-y['recall'])*100;assert d==rr['delta_recall_pp'];delta.append(d)
                assert abs(sum(delta)/len(delta)-q['mean_delta_recall_pp'])<1e-12
                assert q['worst_damage']==min(q['pairs'],key=lambda x:x['delta_recall_pp'])
    expected={'A_four_budgets.csv':36,'A_per_group.csv':252,'B_onepercent_per_group.csv':63,
              'B_onepercent_table.csv':9,'frozen_workpoints.csv':9,'paired_recall_changes.csv':84}
    for filename,n in expected.items():
        with (root/filename).open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==n
    cpu=read('CPU_VERIFIED.json');assert cpu['checked_points']==99 and cpu['exact_frozen_group_reproduction']==63
    result=dict(sealed_files=len(manifest),scan_points=99,frozen_reproduction_rows=63,
                independent_metric_rows=count,all_hashes_budgets_next_ties_denominators_comparisons_verified=True)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    verify(parser.parse_args().root)
