"""Seal label-blind common support, then evaluate three frozen settings locally."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from phase89_cpu_ap import average_precision_score as cpu_ap

R=Path(__file__).resolve().parents[2]
O=R/'src/research_results/phase118_20261014'
D=R/'data/pixbox_s2_cmix_5036991/phase118_20261014'
U=R/'outputs/phase118/phase118_20261014'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def table(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ap(rows,policy):
    y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rows],bool)
    s=np.array([r[policy] for r in rows],np.float32)
    if not y.any() or not (~y).any():return None
    idx=np.argsort(-s,kind='stable');ss=s[idx];ys=y[idx]
    ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True]);tp=np.cumsum(ys,dtype=np.int64)[ends]
    value=float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1)/y.sum()))
    assert abs(value-cpu_ap(y,s))<1e-12
    return value


def operation(rows,policy):
    y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rows],bool)
    pred=np.array([r[policy]>=.5 for r in rows],bool)
    p,n=int(y.sum()),int((~y).sum());tp,fp=int((y&pred).sum()),int((~y&pred).sum())
    return {'positive':p,'negative':n,'TP':tp,'FP':fp,'recall':tp/p if p else None,
            'precision':tp/(tp+fp) if tp+fp else None,'negative_FP_fraction':fp/n if n else None}


def main():
    lock=read(O/'EVALUATION_LOCK.json')
    for p,h in lock['source_sha256'].items():assert sha(R/p)==h,p
    receipt=read(U/'SCORES_SEALED.json');allpoints={p:{} for p in ('O20','C20','H10')};hashes={}
    pids=[r['product_id'] for r in read(R/'src/research_results/phase104_20261013/object_manifest.json')['products']]
    old_seal=read(R/'outputs/phase112/phase112_20261014/SCORES_SEALED.json')
    for policy in allpoints:
        for pid in pids:
            path=(R/'data/pixbox_s2_cmix_5036991/phase112_20261014/omni'/pid/'UNLABELLED_POINT_SCORES.json') if policy=='O20' else D/policy/pid/'UNLABELLED_POINT_SCORES.json'
            expected=old_seal['score_sha256'][pid] if policy=='O20' else receipt['score_sha256'][policy+'/'+pid]
            assert sha(path)==expected;hashes[policy+'/'+pid]=expected
            for row in read(path)['points']:
                assert row['point_id'] not in allpoints[policy];allpoints[policy][row['point_id']]=row
        assert len(allpoints[policy])==9500
    rows=[];loss=collections.defaultdict(lambda:collections.Counter())
    for point_id,c in allpoints['C20'].items():
        h=allpoints['H10'][point_id];o=allpoints['O20'][point_id]
        for q in (o,h):
            for key in ('product_id','row20','col20'):assert str(q[key])==str(c[key])
        for key in ('row10','col10','group_id'):assert h[key]==c[key]
        ok=all(q['valid'] and q['coverage'] and q['author_valid'] for q in (o,c,h))
        entry={k:c[k] for k in ('point_id','product_id','group_id','row10','col10','row20','col20')}
        entry.update(common_valid=ok,**{p:q['shadow_confidence'] for p,q in (('O20',o),('C20',c),('H10',h))})
        if ok:assert all(np.isfinite(entry[p]) for p in allpoints)
        rows.append(entry);l=loss[c['group_id']];l['original_points']+=1;l['common_valid']+=int(ok)
        for p,q in (('O20',o),('C20',c),('H10',h)):l[p+'_invalid']+=int(not q['valid'])
    # Freeze by observation/coverage only, before any reference semantics are loaded.
    write(D/'COMMON_SUPPORT_PRIVATE.json',{'rows':[{'point_id':r['point_id'],'common_valid':r['common_valid']} for r in rows]})
    write(O/'COMMON_SUPPORT_LOCK.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'before_reference_join':True,'original_points':9500,'common_valid':sum(r['common_valid'] for r in rows),
          'point_mask_sha256':sha(D/'COMMON_SUPPORT_PRIVATE.json'),'all_score_sha256':hashes,
          'only_Omni_original_three_channels_required':True,'no_extra_Source_six_band_support_filter':True})
    table(O/'support_losses.csv',[{'group_id':g,**{k:c[k] for k in ('original_points','common_valid','O20_invalid','C20_invalid','H10_invalid')}} for g,c in sorted(loss.items())])
    refs_path=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert sha(refs_path)==lock['original_reference_sha256']
    refs={q['point_ids'][0]:q for q in read(refs_path)};assert set(refs)==set(allpoints['O20'])
    groups=collections.defaultdict(list);joined=[]
    for r in rows:
        ref=refs[r['point_id']]
        assert all(str(r[k])==str(ref[k]) for k in ('product_id','group_id','row20','col20'))
        r=dict(r,reference=ref['reference']);joined.append(r)
        groups[r['group_id']]  # retain every original group even with zero common support
        if r['common_valid'] and r['reference']!='ignore':groups[r['group_id']].append(r)
    table(D/'JOINED_REFERENCE_SCORES_PRIVATE.csv',joined)
    group_rows=[]
    for gid,rr in sorted(groups.items()):
        pos=sum(r['reference']=='positive_ground_cloud_shadow' for r in rr);neg=len(rr)-pos
        d={'group_id':gid,'tiles':','.join(sorted({r['product_id'].split('_T')[1][:5] for r in rows if r['group_id']==gid})),
           'positive':pos,'negative':neg,'sparse_support':bool(pos>0 and (pos<5 or neg<20))}
        for policy in allpoints:
            d[policy+'_AP']=ap(rr,policy)
            d.update({policy+'_'+k:v for k,v in operation(rr,policy).items()})
        for a,b in (('H10','C20'),('C20','O20'),('H10','O20')):
            d[a+'_minus_'+b+'_AP_pp']=(d[a+'_AP']-d[b+'_AP'])*100 if d[a+'_AP'] is not None else None
        group_rows.append(d)
    table(O/'per_group_results.csv',group_rows)
    allrr=[r for rr in groups.values() for r in rr];summary=[]
    for policy in allpoints:
        values=[r[policy+'_AP'] for r in group_rows if r[policy+'_AP'] is not None]
        recalls=[r[policy+'_recall'] for r in group_rows if r[policy+'_recall'] is not None]
        fps=[r[policy+'_negative_FP_fraction'] for r in group_rows if r[policy+'_negative_FP_fraction'] is not None]
        d={'setting':policy,'AP_groups':len(values),'macro_AP_pp':float(np.mean(values))*100 if values else None,
           'pooled_AP_pp':ap(allrr,policy)*100 if ap(allrr,policy) is not None else None,
           'macro_recall_pp':float(np.mean(recalls))*100 if recalls else None,
           'macro_negative_FP_pp':float(np.mean(fps))*100 if fps else None,
           'worst_group_negative_FP_pp':max(fps)*100 if fps else None}
        d.update({'pooled_'+k:v for k,v in operation(allrr,policy).items()});summary.append(d)
    table(O/'three_setting_summary.csv',summary)
    comparisons=[]
    for a,b in (('H10','C20'),('C20','O20'),('H10','O20')):
        key=a+'_minus_'+b+'_AP_pp';eligible=[r for r in group_rows if r[key] is not None]
        delta=np.array([r[key] for r in eligible],float)
        result={'comparison':a+'-'+b,'primary':a=='H10' and b=='C20','n_related_groups':len(delta),
                'mean_delta_AP_pp':float(delta.mean()) if len(delta) else None,'CI95_pp':None,
                'positive_direction_groups':int((delta>0).sum()),'negative_direction_groups':int((delta<0).sum()),
                'exact_zero_groups':int((delta==0).sum()),'seed':118014,'resamples':10000,
                'no_minimum_effect_or_significance_pass_gate':True,'unit':'related group, not pixels or settings'}
        if len(delta)>=2:
            idx=np.random.default_rng(118014).integers(len(delta),size=(10000,len(delta)))
            bootstrap=delta[idx].mean(1);scalar=np.array([sum(float(delta[i]) for i in row)/len(delta) for row in idx])
            assert np.max(np.abs(bootstrap-scalar))<1e-10
            result['CI95_pp']=np.quantile(bootstrap,[.025,.975]).tolist()
        comparisons.append(result)
    write(O/'FIXED_COMPARISONS.json',{'rows':comparisons,'interpretation':'exploratory effect/uncertainty/cost; no automatic closure for CI crossing0'})
    write(O/'CPU_METRIC_ACCEPTANCE.json',{'status':'passed','independent_tied_AP_all_groups_and_pool':True,
          'paired_group_bootstrap_scalar_passed':True,'common_support_sealed_before_original_reference_join':True,
          'sparse_groups_retained':True,'new_CPU_model_calls':0,'no_threshold_scan_or_reference_changes':True,
          'evaluated_positive':sum(r['reference']=='positive_ground_cloud_shadow' for r in allrr),
          'evaluated_negative':sum(r['reference']!='positive_ground_cloud_shadow' for r in allrr),
          'ignore_on_common_support':sum(r['common_valid'] and r['reference']=='ignore' for r in joined)})
    for name in ('COMMON_SUPPORT_LOCK.json','support_losses.csv','per_group_results.csv','three_setting_summary.csv','FIXED_COMPARISONS.json','CPU_METRIC_ACCEPTANCE.json'):
        assert not (U/name).exists();(U/name).write_bytes((O/name).read_bytes())
    print(json.dumps({'summary':summary,'comparisons':comparisons},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
