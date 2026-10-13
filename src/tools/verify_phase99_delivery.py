"""Offline manifest, integer counts, boundaries and fixed Phase99 decision."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def verify(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    digest=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    manifest=read('DELIVERY_SHA256.json')
    for n,h in manifest.items():assert digest(n)==h,n
    lock=read('input_lock.json');report=read('phase99_report.json');cpu=read('CPU_VERIFIED.json')
    close=lambda a,b: a is b if a is None or b is None else abs(a-b)<1e-12
    assert lock['models_included']==['G7_100','SourceS_100','MIN']
    assert len(lock['calibration'])==53 and len(lock['development'])==32
    assert not {a['group_id'] for a in lock['calibration']}&{a['group_id'] for a in lock['development']}
    assert report['oracle_points']==12 and report['frozen_points']==6 and report['bootstrap_resamples']==0
    for n in ['new_fits','new_network_forwards','new_downloads','new_ALCD_pixel_reads']:assert lock[n]==report[n]==0
    assert report['oracle_non_deployable'] and report['Phase98_branches_remain_closed']
    policies=report['policies'];assert len(policies)==18
    rowcount=0
    for domain,count in [('calibration',53),('development',32)]:
        frozen=read(domain+'_frozen_reproduction.json');assert frozen['all_counts_exact']
        for n in lock['models_included']:
            reference=policies[domain+'/'+n+'/frozen']
            assert reference==frozen['policies'][n]
            assert reference['threshold']==lock['old_frozen_thresholds_probability64'][n]
            for kind in ['frozen','O_mean','O_tail']:
                a=policies[domain+'/'+n+'/'+kind];rows=a['rows'];rowcount+=len(rows)
                assert len(rows)==count and len({r['group_id'] for r in rows})==count
                for row in rows:
                    tp=row['true_positive'];fp=row['false_positive'];pos=row['positive'];neg=row['negative']
                    assert 0<=tp<=pos and 0<=fp<=neg and row['threshold']==a['threshold']
                    assert row['surface']+row['cloud']==neg and row['surface_false_positive']+row['cloud_false_positive']==fp
                    assert close(row['recall'],tp/pos if pos else None)
                    assert close(row['fpr'],fp/neg if neg else None)
                    assert close(row['precision'],tp/(tp+fp) if tp+fp else None)
                    assert close(row['shadow_iou'],tp/(pos+fp) if pos+fp else None)
                    if not pos:assert row['ap'] is None and row['recall'] is None
                for field in ['recall','fpr','precision','ap','shadow_iou','surface_fpr','cloud_fpr']:
                    vals=[row[field] for row in rows if row[field] is not None]
                    assert close(a['macro_'+field],float(np.mean(vals)) if vals else None)
                assert a['macro_ap']==reference['macro_ap'] and a['pooled_ap']==reference['pooled_ap']
                assert [row['ap'] for row in rows]==[row['ap'] for row in reference['rows']]
                assert a['max_group_fpr']==max(row['fpr'] for row in rows if row['fpr'] is not None)
                assert a['groups_above_budget']==sum(row['fpr']>.01 for row in rows if row['fpr'] is not None)
                assert a['zero_recall_groups']==[row['product'] for row in rows if row['recall']==0]
                assert a['recall_groups']==sum(row['positive']>0 for row in rows)
                tp=sum(row['true_positive'] for row in rows);fp=sum(row['false_positive'] for row in rows)
                assert a['positive']==sum(row['positive'] for row in rows) and a['negative']==sum(row['negative'] for row in rows)
                assert close(a['pooled_recall'],tp/a['positive'] if a['positive'] else None)
                assert close(a['pooled_fpr'],fp/a['negative'] if a['negative'] else None)
                assert close(a['pooled_precision'],tp/(tp+fp) if tp+fp else None)
                if kind!='frozen':
                    assert a['oracle'] and a['deployable'] is False and a['macro_fpr']<=.01
                    if kind=='O_tail':assert a['max_group_fpr']<=.1
            a=policies[domain+'/'+n+'/O_mean'];b=policies[domain+'/'+n+'/O_tail']
            assert b['threshold']>=a['threshold'] and b['macro_recall']<=a['macro_recall']
    assert rowcount==3*3*(53+32)==765
    points=read('oracle_boundary_audit.json')['points'];assert len(points)==12
    for point in points:
        a=policies[point['domain']+'/'+point['model']+'/'+point['kind']]
        assert point['threshold']==a['threshold']==float(np.nextafter(point['unsafe_negative_boundary'],np.inf))
        assert point['lower_adjacent_threshold']==point['unsafe_negative_boundary']
        assert point['false_positives']==[row['false_positive'] for row in a['rows']]
        assert point['denominators']==[row['negative'] for row in a['rows']]
        assert point['macro_fpr']==a['macro_fpr'] and point['max_group_fpr']==a['max_group_fpr']
        lower=point['lower_adjacent'];rr=[fp/n for fp,n in zip(lower['false_positives'],lower['denominators']) if n]
        assert close(lower['macro_fpr'],float(np.mean(rr))) and lower['max_group_fpr']==max(rr)
        violations=[]
        if lower['macro_fpr']>.01:violations.append('mean_FPR>1%')
        if point['kind']=='O_tail' and lower['max_group_fpr']>.1:violations.append('max_group_FPR>10%')
        assert violations==point['violations'] and violations
        assert [b-a for a,b in zip(point['false_positives'],lower['false_positives'])]==point['boundary_tie_counts']
    assert len(report['comparisons'])==10
    for c in report['comparisons']:
        if c['comparison']=='MIN_minus_SourceS_O_tail':aa,bb='MIN/O_tail','SourceS_100/O_tail'
        elif c['comparison']=='G7_minus_SourceS_O_tail':aa,bb='G7_100/O_tail','SourceS_100/O_tail'
        else:
            name=c['comparison'].removesuffix('_tail_minus_mean');aa,bb=name+'/O_tail',name+'/O_mean'
        a=policies[c['domain']+'/'+aa];b=policies[c['domain']+'/'+bb]
        assert close(c['mean_delta_recall_pp'],100*(a['macro_recall']-b['macro_recall']))
    primary=next(c for c in report['comparisons'] if c['domain']=='development' and c['comparison']=='MIN_minus_SourceS_O_tail')
    assert report['primary']==primary
    decision='存在值得另立校准研究的经验工作点' if primary['mean_delta_recall_pp']>=5 else '本轮未发现达到既定召回增量的全局阈值工作点'
    assert report['decision']==decision
    assert len(cpu['score_probes'])==170
    for row in cpu['score_probes']:assert row['max_error']<=(1e-4 if row['model']=='SourceS_100' else 6e-8)
    scores=read('input_score_lock.json');assert len(scores['rows'])==85 and scores['indices_after']==lock['indices_before']
    return dict(verified=True,files=len(manifest),oracle_points=12,frozen_points=6,per_group_rows=rowcount,
        decision=decision,primary_recall_gain_pp=primary['mean_delta_recall_pp'],no_fits_or_forwards=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().root),ensure_ascii=False,indent=2))
