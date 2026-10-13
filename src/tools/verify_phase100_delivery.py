"""Offline Phase100 SHA, calibration boundaries, group bootstrap and screening."""
import argparse,hashlib,json,csv
from pathlib import Path
import numpy as np

def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert digest(n)==h,n
    lock=read('input_lock.json');th=read('three_T_tail_lock.json');report=read('phase100_report.json');cpu=read('CPU_VERIFIED.json')
    names=['G7_100','SourceS_100','MIN'];close=lambda a,b:a is b if a is None or b is None else abs(a-b)<1e-12
    assert lock['models_included']==names and len(lock['fit'])==100 and len(lock['calibration'])==53 and len(lock['development'])==32
    calids={a['group_id'] for a in lock['calibration']};devids={a['group_id'] for a in lock['development']}
    assert len(calids)==53 and len(devids)==32 and not calids&devids
    for n in ['new_fits','new_network_forwards','new_downloads','new_ALCD_pixel_reads']:assert lock[n]==report[n]==0
    assert lock['new_calibration_thresholds']==report['new_calibration_thresholds']==th['new_calibration_thresholds']==3
    assert lock['bootstrap_seed']==report['bootstrap_seed']==100013 and lock['bootstrap_resamples']==report['bootstrap_resamples']==10000
    assert not lock['development_oracle_used_for_thresholds'] and not report['development_independent_confirmation']
    assert report['Phase98_failure_unchanged'] and th['all_frozen_before_development'] and not th['development_oracle_read_before_lock']
    assert th['input_lock_sha256']==digest('input_lock.json') and th['Phase99_calibration_O_tail_reproduced']
    def interval(got,values):
        x=np.asarray([v for v in values if v is not None],np.float64)
        assert got['seed']==100013 and got['resamples']==10000 and got['n_groups']==len(x)
        assert close(got['mean'],float(x.mean()) if len(x) else None)
        if len(x)>1:
            z=x[np.random.default_rng(100013).integers(len(x),size=(10000,len(x)))].mean(axis=1)
            np.testing.assert_allclose(got['ci95'],np.quantile(z,[.025,.975]),rtol=0,atol=1e-12)
            assert close(got['upper95_one_sided'],float(np.quantile(z,.95)))
        else:assert got['ci95'] is None and got['upper95_one_sided'] is None
    def counts(row):
        pos=row['positive'];neg=row['negative'];tp=row['true_positive'];fp=row['false_positive']
        assert 0<=tp<=pos and 0<=fp<=neg and row['surface']+row['cloud']==neg
        assert row['surface_false_positive']+row['cloud_false_positive']==fp
        for key,value in dict(recall=tp/pos if pos else None,fpr=fp/neg if neg else None,
            precision=tp/(tp+fp) if tp+fp else None,shadow_iou=tp/(pos+fp) if pos+fp else None).items():assert close(row[key],value)
        for key in ['surface','cloud']:assert close(row[key+'_fpr'],row[key+'_false_positive']/row[key] if row[key] else None)
        if not pos:assert row['ap'] is None and row['recall'] is None
    def summary(a,count,ids):
        rows=a['rows'];assert len(rows)==count and {row['group_id'] for row in rows}==ids
        assert not a['oracle']
        for row in rows:counts(row);assert row['threshold']==a['threshold']
        for key in ['ap','recall','precision','fpr','shadow_iou','surface_fpr','cloud_fpr']:
            vals=[row[key] for row in rows if row[key] is not None]
            assert close(a['macro_'+key],float(np.mean(vals)) if vals else None)
            interval(a['intervals'][key],vals)
        interval(a['no_shadow_fpr'],[row['fpr'] for row in rows if not row['positive']])
        assert a['max_group_fpr']==max(row['fpr'] for row in rows if row['fpr'] is not None)
        assert a['groups_above_budget']==sum(row['fpr']>.01 for row in rows if row['fpr'] is not None)
        assert a['zero_recall_groups']==[row['product'] for row in rows if row['recall']==0]
        assert a['recall_groups']==sum(row['positive']>0 for row in rows)
        tp=sum(row['true_positive'] for row in rows);fp=sum(row['false_positive'] for row in rows)
        pos=sum(row['positive'] for row in rows);neg=sum(row['negative'] for row in rows)
        assert a['positive']==pos and a['negative']==neg
        for key,v in dict(recall=tp/pos if pos else None,fpr=fp/neg if neg else None,
            precision=tp/(tp+fp) if tp+fp else None,shadow_iou=tp/(pos+fp) if pos+fp else None).items():assert close(a['pooled_'+key],v)
    for name,t in th['models'].items():
        assert name in names and t['threshold']==lock['calibration_only_reference'][name]
        a=t['calibration'];summary(a,53,calids);b=t['boundary']
        assert a['macro_fpr']<=.01 and a['max_group_fpr']<=.1
        assert b['threshold']==t['threshold']==float(np.nextafter(b['unsafe_negative_boundary'],np.inf))
        assert b['false_positives']==[row['false_positive'] for row in a['rows']]
        assert b['denominators']==[row['negative'] for row in a['rows']]
        assert b['macro_fpr']==a['macro_fpr'] and b['max_group_fpr']==a['max_group_fpr']
        below=b['lower_adjacent'];rr=[fp/n for fp,n in zip(below['false_positives'],below['denominators']) if n]
        assert close(below['macro_fpr'],float(np.mean(rr))) and below['max_group_fpr']==max(rr)
        assert below['macro_fpr']>.01 or below['max_group_fpr']>.1
        assert [b-a for a,b in zip(b['false_positives'],below['false_positives'])]==b['boundary_tie_counts']
        assert len(t['negative_score_sha256'])==53
    policies=report['policies'];assert set(policies)=={n+'/'+k for n in names for k in ['old','T_tail']}
    for n in names:
        for kind in ['old','T_tail']:
            a=policies[n+'/'+kind];summary(a,32,devids)
            assert a['threshold']==(lock['old_frozen_thresholds'][n] if kind=='old' else th['models'][n]['threshold'])
        a=policies[n+'/old'];b=policies[n+'/T_tail']
        assert a['macro_ap']==b['macro_ap'] and a['pooled_ap']==b['pooled_ap']
        assert [row['ap'] for row in a['rows']]==[row['ap'] for row in b['rows']]
    defs=[('MIN_T_tail_minus_SourceS_T_tail','MIN/T_tail','SourceS_100/T_tail'),('G7_T_tail_minus_SourceS_T_tail','G7_100/T_tail','SourceS_100/T_tail')]
    defs +=[(n+'_tail_minus_old',n+'/T_tail',n+'/old') for n in names]
    assert set(report['comparisons'])=={name for name,_,_ in defs}
    for name,aa,bb in defs:
        c=report['comparisons'][name];assert len(c['pairs'])==32
        for x,y,z in zip(policies[aa]['rows'],policies[bb]['rows'],c['pairs']):
            assert x['group_id']==y['group_id']==z['group_id']
            for k in ['ap','recall','fpr','precision','shadow_iou','surface_fpr','cloud_fpr']:
                assert close(z['delta_'+k+'_pp'],100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None)
        for k,v in c['intervals_pp'].items():interval(v,[z['delta_'+k+'_pp'] for z in c['pairs']])
    base=policies['SourceS_100/T_tail'];retained=[]
    for n in ['G7_100','MIN']:
        a=policies[n+'/T_tail'];gain=100*(a['macro_recall']-base['macro_recall']);apgain=100*(a['macro_ap']-base['macro_ap'])
        criteria=dict(macro_fpr_at_most_1percent=a['macro_fpr']<=.01,max_group_fpr_at_most_10percent=a['max_group_fpr']<=.1,
            recall_gain_at_least_5pp=gain>=5,macro_AP_loss_at_most_1pp=apgain>=-1)
        got=report['screening'][n];assert got['criteria']==criteria and got['all_four_pass']==all(criteria.values())
        assert close(got['delta_recall_pp'],gain) and close(got['delta_AP_pp'],apgain)
        if all(criteria.values()):retained.append(n)
    assert report['retained_candidates']==retained and report['decision']==('eligible_for_separate_stability_protocol' if retained else 'end_global_threshold_repair_route')
    assert report['SourceS_budget']==dict(macro_point_at_most_1percent=base['macro_fpr']<=.01,max_group_at_most_10percent=base['max_group_fpr']<=.1,
        macro_upper95_at_most_1percent=base['intervals']['fpr']['upper95_one_sided']<=.01)
    scores=read('input_score_lock.json');assert len(scores['rows'])==85 and scores['indices_after']==lock['indices_before']
    assert scores['three_threshold_lock_sha256']==digest('three_T_tail_lock.json')
    assert len(cpu['score_probes'])==170 and cpu['development_oracle_read_only_after_all_six_evaluations']
    for row in cpu['score_probes']:assert row['max_error']<=(1e-4 if row['model']=='SourceS_100' else 6e-8)
    assert len(read('T31TGF_T02LKH.json')['rows'])==12
    oc=read('oracle_comparison.json');assert oc['after_six_frozen_evaluations'] and len(oc['rows'])==3
    for row in oc['rows']:
        a=policies[row['model']+'/T_tail'];assert row['calibration_T_tail']==a['threshold']
        assert row['T_tail_macro_recall']==a['macro_recall'] and row['T_tail_macro_fpr']==a['macro_fpr']
        assert row['labels_used_to_select_oracle'] and row['not_achievable_calibration_gain']
    for file,count in [('six_strategy_totals.csv',6),('per_group.csv',192),('paired_changes.csv',160),('oracle_comparison.csv',3)]:
        with (root/file).open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==count,file
    return dict(verified=True,payload_files=len(manifest),thresholds=3,development_policies=6,development_rows=192,paired_rows=160,
        calibration_rows=159,retained_candidates=retained,decision=report['decision'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().root),ensure_ascii=False,indent=2))
