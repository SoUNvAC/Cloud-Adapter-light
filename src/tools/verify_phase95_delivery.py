"""Independent numpy-only Phase95 small-result acceptance; no scoring or fits."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items(): assert digest(n)==h,n
    lock=read('input_lock.json'); tl=read('four_new_threshold_lock.json'); report=read('phase95_report.json'); cpu=read('CPU_VERIFIED.json')
    assert lock['new_fits']==report['new_fits']==report['new_network_forwards']==report['new_downloads']==0
    assert report['new_thresholds']==tl['new_thresholds']==4 and tl['frozen_before_new_evaluation']
    assert tl['input_lock_sha256']==digest('input_lock.json')
    assert tl['calibration_support_lock_sha256']==digest('calibration_support_lock.json')
    assert len(lock['calibration'])==53 and len(lock['fit'])==100
    ordered=sorted(lock['calibration']+lock['fit'],key=lambda r:(hashlib.sha256(('PHASE94_SPLIT_V1|'+r['group_id']).encode()).hexdigest(),r['group_id']))
    assert ordered[:100]==lock['fit'] and ordered[100:]==lock['calibration']
    assert read('input_score_lock.json')['indices_after']==lock['indices_before']
    assert read('input_score_lock.json')['four_threshold_lock_sha256']==digest('four_new_threshold_lock.json')
    def close(a,b):
        if b is None: assert a is None
        else: assert a is not None and abs(a-b)<1e-12,(a,b)
    def interval(it,values):
        close(it['mean'],float(np.mean(values)) if values else None)
        assert it['n_groups']==len(values) and it['seed']==95012 and it['resamples']==10000
        if len(values)>1:
            x=np.array(values); b=x[np.random.default_rng(95012).integers(len(x),size=(10000,len(x)))].mean(1)
            np.testing.assert_allclose(it['ci95'],np.quantile(b,[.025,.975]),rtol=0,atol=1e-12)
            close(it['upper95_one_sided'],float(np.quantile(b,.95)))
    support=read('calibration_support_lock.json')['rows']; assert len(support)==53
    for r in support: assert r['negative_pixels']==9*r['negative_blocks'] and r['eligible']==bool(r['negative_blocks'])
    for n,types in tl['models'].items():
        for t in ['T20_same','T60_same']:
            v=types[t]; assert len(v['rows'])==53 and v['dtype']==('float64' if t=='T60_same' else 'float32')
            assert float.fromhex(v['threshold_hex'])==v['threshold']
            rows=[a for a in v['rows'] if a['negative_count']]
            rate=float(np.mean([a['false_positive']/a['negative_count'] for a in rows]))
            close(v['macro_fpr'],rate); assert rate<=.01 and (v['next_macro_fpr'] is None or v['next_macro_fpr']>.01)
            assert len(rows)==v['negative_groups']; close(v['group_weight'],1/len(rows))
            for a,r in zip(v['rows'],support): assert a['negative_count']==r['negative_pixels' if t=='T20_same' else 'negative_blocks']
            expected=v['threshold'] if n=='G7_100' or t=='T60_same' else float(np.exp(-np.logaddexp(0.,-v['threshold'])))
            close(v['probability_threshold'],expected)
        assert [a['negative_count'] for a in types['T20_same']['rows']]==[9*a['negative_count'] for a in types['T60_same']['rows']]
        assert types['T20_full']['reused_phase94']
    for domain,total in [('calibration',53),('development_same60',32)]:
        rows=[a for a in report['support_retention'] if a['domain']==domain]
        assert len(rows)==3*total
        for a in rows:
            assert a['retained_pixels']==9*a['retained_blocks']<=a['cropped_pixels']<=a['full_valid_pixels']
            close(a['retention_of_full'],a['retained_pixels']/a['full_valid_pixels'] if a['full_valid_pixels'] else None)
            close(a['retention_of_crop'],a['retained_pixels']/a['cropped_pixels'] if a['cropped_pixels'] else None)
    count=0
    for domain,v in report['domains'].items():
        expected_groups=dict(development_same60=32,original_main=9,original_common=6,revised_common=6)[domain]
        assert len(v['policies'])==6 and v['same_model_AP_exact_invariant']
        for name,s in v['policies'].items():
            assert len(s['rows'])==expected_groups and len({a['group_id'] for a in s['rows']})==expected_groups
            n,t=name.rsplit('_T',1); close(s['threshold'],tl['models'][n]['T'+t]['probability_threshold'])
            for a in [*s['rows'],s['pooled']]:
                tp=a['true_positive']; fp=a['false_positive']; pos=a['positive']; neg=a['negative']
                assert fp==a['surface_false_positive']+a['cloud_false_positive'] and neg==a['surface']+a['cloud']
                close(a['recall'],tp/pos if pos else None); close(a['fpr'],fp/neg if neg else None)
                close(a['precision'],tp/(tp+fp) if tp+fp else None); close(a['shadow_iou'],tp/(pos+fp) if pos+fp else None)
                close(a['surface_fpr'],a['surface_false_positive']/a['surface'] if a['surface'] else None)
                close(a['cloud_fpr'],a['cloud_false_positive']/a['cloud'] if a['cloud'] else None)
                if not pos: assert a['ap'] is None and a['recall'] is None
                count+=1
            for k,it in s['macro'].items(): interval(it,[a[k] for a in s['rows'] if a[k] is not None])
            interval(s['no_shadow_macro_fpr'],[a['fpr'] for a in s['rows'] if not a['positive'] and a['fpr'] is not None])
            close(s['max_group_fpr'],max((a['fpr'] for a in s['rows'] if a['fpr'] is not None),default=None))
            assert s['groups_above_1percent']==sum(a['fpr'] is not None and a['fpr']>.01 for a in s['rows'])
            assert s['shadow_groups']==sum(a['positive']>0 for a in s['rows'])
            assert s['zero_recall_groups']==[a['product'] for a in s['rows'] if a['recall']==0]
            for k in ['true_positive','false_positive','positive','negative','surface','cloud','surface_false_positive','cloud_false_positive']: assert s['pooled'][k]==sum(a[k] for a in s['rows'])
        for n in tl['models']:
            policies=[v['policies'][n+'_'+t] for t in ['T20_full','T20_same','T60_same']]
            assert all(a['macro']['ap']==policies[0]['macro']['ap'] and a['pooled']['ap']==policies[0]['pooled']['ap'] for a in policies)
        for name,c in v['comparisons'].items():
            left,right=name.split('_minus_'); model,tail=left.rsplit('_T',1)
            a=v['policies'][left]['rows']; b=v['policies'][model+'_'+right]['rows']
            assert len(c['pairs'])==expected_groups
            for x,y,z in zip(a,b,c['pairs']):
                assert x['group_id']==y['group_id']==z['group_id']
                for k in ['ap','fpr','recall','precision','shadow_iou']: close(z['delta_'+k+'_pp'],100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None)
            for k in ['ap','fpr','recall','precision','shadow_iou']: interval(c['delta_'+k+'_pp'],[a['delta_'+k+'_pp'] for a in c['pairs'] if a['delta_'+k+'_pp'] is not None])
    assert len(cpu['phase94_reproduction'])==2*(32+9+6+6)
    assert all(a['metric_row_exact'] and a['score_and_truth_sha_exact'] for a in cpu['phase94_reproduction'])
    assert len(cpu['rows'])==818
    assert all(a['max_error']<=(6e-8 if a['model']=='G7_100' else 1e-4) and a.get('mean9_error',0)<=2e-15 for a in cpu['rows'])
    assert cpu['tests']['float64_adjacent_tie_test'] and cpu['all_mean9_and_decision_checks']
    return dict(verified=True,sealed_files=len(manifest),independent_metric_rows=count,score_probes=len(cpu['rows']),phase94_reproduced_rows=len(cpu['phase94_reproduction']),new_fits=0,new_thresholds=4)


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(a.parse_args().root),indent=2))
