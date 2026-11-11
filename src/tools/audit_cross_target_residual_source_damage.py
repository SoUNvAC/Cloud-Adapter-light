"""Compare frozen L8/SPARCS source ROI confusions; no pixel or model access."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/cross_target_residual_damage_20261024'
locks={}


def read(relative):
    p=ROOT/relative;locks[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def miou(c):
    denom=[sum(c[i])+sum(row[i] for row in c)-c[i][i] for i in range(3)]
    assert all(v>0 for v in denom)
    return 100*sum(c[i][i]/denom[i] for i in range(3))/3


records={}
for target,phase,date in [('L8',222,'20261024'),('SPARCS',217,'20261023')]:
    for r in read(f'outputs/phase{phase}/phase{phase}_{date}/PER_GROUP.csv'):
        if r['pool']=='source':
            records[target,r['seed'],r['arm'],float(r['alpha']),r['group']]=json.loads(r['confusion'])
l8rule=read('outputs/phase222/phase222_20261024/JOINT_RULE.csv')
sprule=read('outputs/phase217/phase217_20261023/JOINT_RULE.csv')
rois=sorted(k[4] for k in records if k[:4]==('L8','21701','C',0.))
assert len(rois)==107
baseline={roi:records['L8','21701','C',0.,roi] for roi in rois}
checks=0
for (target,seed,arm,alpha,roi),c in records.items():
    assert sum(map(sum,c))==sum(map(sum,baseline[roi]))
    assert list(map(sum,c))==list(map(sum,baseline[roi]))
    if alpha==0:assert c==baseline[roi]
    checks+=3 if alpha==0 else 2

rows=[];flows=[];tails={};summary=[]
for seed in ['21701','21702']:
    al8=float(next(r['alpha'] for r in l8rule if r['seed']==seed and r['arm']=='D' and r['budget_pp']=='1'))
    asp=float(next(r['D_alpha'] for r in sprule if r['seed']==seed and r['context']=='B1'))
    assert al8==.25
    for target,alpha,context in [('L8',al8,'selected_B1'),('SPARCS',asp,'selected_B1'),('L8',.25,'matched_alpha025'),('SPARCS',.25,'matched_alpha025')]:
        local=[]
        for roi in rois:
            src=baseline[roi];c=records[target,seed,'D',alpha,roi]
            loss=miou(src)-miou(c)
            differences=[(c[i][j]-src[i][j],f'{a}->{b}') for i,a in enumerate(['Surface','Cloud','Shadow']) for j,b in enumerate(['Surface','Cloud','Shadow']) if i!=j]
            dominant=max(differences)
            rr=dict(target=target,seed=seed,context=context,alpha=alpha,ROI=roi,
                    Source_mIoU=miou(src),D_mIoU=miou(c),loss_pp=loss,
                    net_errors=sum(c[i][j]-src[i][j] for i in range(3) for j in range(3) if i!=j),
                    largest_net_added_error_flow=dominant[1],largest_net_added_error_pixels=dominant[0])
            local.append(rr)
            for i,a in enumerate(['Surface','Cloud','Shadow']):
                for j,b in enumerate(['Surface','Cloud','Shadow']):
                    if i!=j:flows.append(dict(target=target,seed=seed,context=context,alpha=alpha,ROI=roi,flow=f'{a}->{b}',Source_pixels=src[i][j],D_pixels=c[i][j],net_added_pixels=c[i][j]-src[i][j]))
        local.sort(key=lambda r:(-r['loss_pp'],r['ROI']))
        for rank,r in enumerate(local,1):r['rank']=rank;r['in_T11']=rank<=11
        rows.extend(local);tails[target,seed,context]={r['ROI'] for r in local[:11]}
        summary.append(dict(target=target,seed=seed,context=context,alpha=alpha,
                            T11_pp=sum(r['loss_pp'] for r in local[:11])/11,
                            worst_ROI=local[0]['ROI'],worst_loss_pp=local[0]['loss_pp'],
                            loss_gt5_ROIs=';'.join(r['ROI'] for r in local if r['loss_pp']>5),
                            loss_gt5_count=sum(r['loss_pp']>5 for r in local)))
overlap=[]
for context in ['selected_B1','matched_alpha025']:
    for seed in ['21701','21702']:
        a,b=tails['L8',seed,context],tails['SPARCS',seed,context]
        overlap.append(dict(seed=seed,context=context,top11_overlap=len(a&b),shared_ROIs=';'.join(sorted(a&b)),
                            union_count=len(a|b),note='Selected alphas may differ; matched-alpha comparison reported separately.'))
focus={r['worst_ROI'] for r in summary if r['target']=='L8' and r['context']=='selected_B1'}
focus_rows=[r for r in rows if r['ROI'] in focus]
OUT.mkdir(parents=True,exist_ok=True)
save('ALL_ROI_SOURCE_DAMAGE.csv',rows)
save('ERROR_FLOW_NET_CHANGES.csv',flows)
save('CASE_SUMMARY.csv',summary)
save('TAIL_OVERLAP.csv',overlap)
save('L8_SELECTED_WORST_ROI_CROSS_TARGET.csv',focus_rows)
save('FOCUS_ERROR_FLOWS.csv',[r for r in flows if r['ROI'] in focus])
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in locks.items())
audit=dict(input_hashes_unchanged=True,checks=checks,training=0,forwards=0,new_bootstrap=0,
           inputs=[dict(path=str(p.relative_to(ROOT)),sha256=h) for p,h in locks.items()],
           limitations=['Selected-alpha confusions give net flows, not direct Source-to-D same-pixel transitions.',
                        'Existing L8 direct correctness counts are alpha=1 and cannot explain selected alpha=.25 directly.',
                        'Cross-target recurrence is on the same source ROIs, not independent scene replication.',
                        'ROI identifiers do not establish land-cover type or physical cause.',
                        'B1 is displayed as a unique workpoint; B3/B5 select the same alphas.'])
(OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2),encoding='utf-8')
print(json.dumps(dict(summary=summary,overlap=overlap,focus=focus_rows,output=str(OUT)),indent=2))
