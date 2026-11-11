"""Describe frozen lambda/alpha candidates. Never select deployment using test."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/phase224/phase224_20261024'
OUT=ROOT/'outputs/research_audits/lambda_alpha_finite_frontier_20261024'
locks={}


def read(name):
    p=BASE/name;locks[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def dominates(a,b):
    da=[a['test_mIoU']-b['test_mIoU'],b['pool_loss_pp']-a['pool_loss_pp'],b['T11_pp']-a['T11_pp']]
    return all(v>=0 for v in da) and any(v>0 for v in da)


raw=read('TABLE_C_ALL_CANDIDATES.csv')
per=read('PER_GROUP.csv')
joint=read('JOINT_RULE.csv')
source_rows={(r['target'],r['seed'],r['arm'],float(r['alpha']),r['group']):r for r in per if r['pool']=='source'}
records={};checks=0
for r in raw:
    key=(r['target'],r['seed'],r['arm'],float(r['alpha']))
    records[key]={**r,**{k:float(r[k]) for k in ['alpha','pool_loss_pp','T11_pp','worst_ROI_loss_pp','test_mIoU']}}
    rr=[v for k,v in source_rows.items() if k[:4]==key]
    assert len(rr)==107
    losses=sorted([float(v['loss_pp']) for v in rr],reverse=True)
    assert abs(sum(losses[:11])/11-float(r['T11_pp']))<1e-10
    assert abs(losses[0]-float(r['worst_ROI_loss_pp']))<1e-10
    checks+=2

out=[];pairs=[];frontgroups=[]
for target in ['SPARCS','L8']:
    for seed in ['21701','21702']:
        points=[dict(records[target,seed,'D',0.],candidate='Source',arm='Source')]
        points += [dict(records[target,seed,arm,a],candidate=f'{arm}_a{a:g}') for arm in ['D','H'] for a in [.25,.5,.75,1.]]
        for x in points:
            ds=[y['candidate'] for y in points if dominates(y,x)]
            rr=dict(target=target,seed=seed,candidate=x['candidate'],arm=x['arm'],alpha=x['alpha'],
                    target_test_mIoU=x['test_mIoU'],gain_vs_Source_pp=x['test_mIoU']-points[0]['test_mIoU'],
                    pool_loss_pp=x['pool_loss_pp'],T11_pp=x['T11_pp'],worst_ROI=x['worst_ROI'],
                    worst_loss_pp=x['worst_ROI_loss_pp'],ROI_loss_gt5=x['ROI_loss_gt5'],
                    on_finite_frontier=not ds,dominated_by=';'.join(ds))
            out.append(rr)
            for y in points:
                if dominates(y,x):pairs.append(dict(target=target,seed=seed,dominator=y['candidate'],dominated=x['candidate'],
                                                   target_gain_pp=y['test_mIoU']-x['test_mIoU'],
                                                   pool_loss_change_pp=y['pool_loss_pp']-x['pool_loss_pp'],T11_change_pp=y['T11_pp']-x['T11_pp']))
            if not ds and x['arm']!='Source':
                ranks=sorted([v for k,v in source_rows.items() if k[:4]==(target,seed,x['arm'],x['alpha'])],key=lambda v:(-float(v['loss_pp']),v['group']))
                for rank,v in enumerate(ranks[:11],1):
                    src=source_rows[target,seed,'D',0.,v['group']]
                    cm,base=json.loads(v['confusion']),json.loads(src['confusion'])
                    flows=[(cm[i][j]-base[i][j],f'{a}->{b}') for i,a in enumerate(['Surface','Cloud','Shadow']) for j,b in enumerate(['Surface','Cloud','Shadow']) if i!=j]
                    change,flow=max(flows)
                    frontgroups.append(dict(target=target,seed=seed,candidate=x['candidate'],rank=rank,ROI=v['group'],loss_pp=v['loss_pp'],dominant_net_added_error=flow,net_added_pixels=change))

paired=[]
for target in ['SPARCS','L8']:
    for arm in ['D','H']:
        for a in [.25,.5,.75,1.]:
            candidates=[r for r in out if r['target']==target and r['arm']==arm and r['alpha']==a]
            assert len(candidates)==2
            paired.append(dict(target=target,arm=arm,alpha=a,
                               mean_test_mIoU=sum(r['target_test_mIoU'] for r in candidates)/2,
                               mean_gain_vs_Source_pp=sum(r['gain_vs_Source_pp'] for r in candidates)/2,
                               mean_pool_loss_pp=sum(r['pool_loss_pp'] for r in candidates)/2,
                               mean_T11_pp=sum(r['T11_pp'] for r in candidates)/2,
                               nondominated_in_both_seeds=all(r['on_finite_frontier'] for r in candidates),
                               seed21701_on_frontier=next(r['on_finite_frontier'] for r in candidates if r['seed']=='21701'),
                               seed21702_on_frontier=next(r['on_finite_frontier'] for r in candidates if r['seed']=='21702')))
jointflags=[]
for r in joint:
    match=next(x for x in out if x['target']==r['target'] and x['seed']==r['seed'] and
               (x['candidate']=='Source' if float(r['alpha'])==0 else x['arm']==r['arm'] and x['alpha']==float(r['alpha'])))
    jointflags.append(dict(target=r['target'],seed=r['seed'],arm=r['arm'],budget=r['budget_pp'],alpha=r['alpha'],
                           on_finite_frontier=match['on_finite_frontier'],dominated_by=match['dominated_by']))
OUT.mkdir(parents=True,exist_ok=True)
save('ALL_D_H_CANDIDATES_AND_FRONTIER.csv',out)
save('TWO_SEED_CANDIDATE_SUMMARY.csv',paired)
save('ALL_DOMINANCE_PAIRS.csv',pairs)
save('FRONTIER_TOP11_ROIS.csv',frontgroups)
save('EXISTING_JOINT_SELECTION_FRONTIER_STATUS.csv',jointflags)
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in locks.items())
audit=dict(checks=checks,inputs=[dict(path=str(p.relative_to(ROOT)),sha256=h) for p,h in locks.items()],
           inputs_unchanged=True,new_training=0,new_forwards=0,new_intervals=0,
           definition='Finite point dominance: target test mIoU >=, source pooled loss <=, source T11 <=; at least one strict.',
           limitations=['Only lambda 1/4 and fixed five alpha values; not a continuous or optimal frontier.',
                        'Already-viewed test used for descriptive comparison, not deployment selection.',
                        'Same source ROIs reused across target domains; no new independent sample.',
                        'No confidence interval for frontier membership; seed agreement is descriptive.',
                        'Worst single ROI and class damage are reported but not part of this three-coordinate dominance definition.'])
(OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),frontier=[r for r in out if r['on_finite_frontier']],summary=paired,
                      dominated_selected=[r for r in jointflags if not r['on_finite_frontier']]),indent=2))
