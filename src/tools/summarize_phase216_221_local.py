"""Consolidate frozen 216--221 tables locally; no new performance or CI computation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/research_audits/phase216_221_synthesis_20261024'
locks={}


def read(relative):
    p=ROOT/relative
    locks[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def save(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


raw=read('outputs/phase221/phase221_20261024/ALL_CANDIDATES.csv')
initial=read('outputs/phase216/phase216_20261023/ALL_CANDIDATES.csv')
old=read('outputs/phase217/phase217_20261023/ALL_CANDIDATES.csv')
oldg=read('outputs/phase219/phase219_20261023/ALL_CANDIDATES.csv')
rule=read('outputs/phase221/phase221_20261024/JOINT_RULE.csv')
checks=0
fields=['pool_loss_pp','T11_pp','worst_ROI_loss_pp','test_mIoU','val_mIoU']
for earlier in [old,oldg]:
    lookup={(r['seed'],r['arm'],float(r['alpha'])):r for r in earlier}
    for r in raw:
        key=(r['seed'],r['arm'],float(r['alpha']))
        if key in lookup:
            for k in fields:
                assert abs(float(r[k])-float(lookup[key][k]))<1e-10
                checks+=1
assert len(raw)==60
source_test=float(raw[0]['test_mIoU'])
assert all(float(r['test_mIoU'])==source_test for r in raw if float(r['alpha'])==0)
summary=[]
for arm in ['C','D','G','S','N','P']:
    for alpha in [0,.25,.5,.75,1]:
        rr=[r for r in raw if r['arm']==arm and float(r['alpha'])==alpha]
        assert len(rr)==2 and {r['seed'] for r in rr}=={'21701','21702'}
        out=dict(arm=arm,alpha=alpha,seeds='21701;21702')
        for k in fields+['ROI_loss_gt5']:
            out[k+'_mean']=sum(float(r[k]) for r in rr)/2
            out[k+'_seed21701']=float(next(r[k] for r in rr if r['seed']=='21701'))
            out[k+'_seed21702']=float(next(r[k] for r in rr if r['seed']=='21702'))
        out['target_gain_vs_Source_pp']=out['test_mIoU_mean']-source_test
        summary.append(out)

selected=[]
for seed in ['21701','21702']:
    for arm in ['C','D','G','S','N','P']:
        for b in [1,3,5]:
            eligible=[r for r in raw if r['seed']==seed and r['arm']==arm and float(r['pool_loss_pp'])<=b and float(r['T11_pp'])<=5]
            pick=max(eligible,key=lambda r:float(r['alpha']))
            prior=next(r for r in rule if r['seed']==seed and r['arm']==arm and float(r['budget_pp'])==b)
            assert float(prior['alpha'])==float(pick['alpha']);checks+=1
            selected.append(dict(seed=seed,arm=arm,budget_pp=b,alpha=float(pick['alpha']),
                                 test_mIoU=float(pick['test_mIoU']),target_gain_vs_Source_pp=float(pick['test_mIoU'])-source_test,
                                 pool_loss_pp=float(pick['pool_loss_pp']),T11_pp=float(pick['T11_pp'])))

# A descriptive finite-grid check: no selection of a model using test results.
dom=[]
for seed in ['21701','21702']:
    for a in ['D','G','S','N','P']:
        for baseline in ['C','D']:
            if a==baseline:continue
            for x in [r for r in raw if r['seed']==seed and r['arm']==a and float(r['alpha'])>0]:
                for y in [r for r in raw if r['seed']==seed and r['arm']==baseline and float(r['alpha'])>0]:
                    dx=float(x['test_mIoU'])-float(y['test_mIoU'])
                    dm=float(x['pool_loss_pp'])-float(y['pool_loss_pp'])
                    dt=float(x['T11_pp'])-float(y['T11_pp'])
                    if dx>=0 and dm<=0 and dt<=0 and (dx>0 or dm<0 or dt<0):
                        dom.append(dict(seed=seed,candidate=a,candidate_alpha=x['alpha'],baseline=baseline,baseline_alpha=y['alpha'],target_difference_pp=dx,pool_loss_difference_pp=dm,T11_difference_pp=dt))
OUT.mkdir(parents=True,exist_ok=True)
save('ALL_ALPHA_TWO_SEED_SUMMARY.csv',summary)
save('ENDPOINT_TWO_SEED_SUMMARY.csv',[r for r in summary if r['alpha']==1])
save('JOINT_RULE_RECHECK.csv',selected)
save('FINITE_THREE_METRIC_POINT_DOMINANCE.csv',dom)
save('PHASE216_INITIAL_SEED_SEPARATE.csv',initial)
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in locks.items())
audit=dict(checks=checks,historical_tables_unchanged=True,inputs=[dict(path=str(p.relative_to(ROOT)),sha256=h) for p,h in locks.items()],
           training=0,forward_calls=0,new_CI_draws=0,
           limitations=['21601 remains separate; only 21701/21702 averaged.',
                        'Mean metrics are not metrics of ensembled predictions.',
                        'Finite point dominance is descriptive, uses already-viewed test and does not choose a deployment model.',
                        '107 source ROIs are not certified geographically independent.',
                        'The target test has been viewed repeatedly; no new independent confirmation.'])
(OUT/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),checks=checks,endpoints=[{k:r[k] for k in ['arm','test_mIoU_mean','target_gain_vs_Source_pp','pool_loss_pp_mean','T11_pp_mean','ROI_loss_gt5_mean']} for r in summary if r['alpha']==1],new_methods_dominating_D=[r for r in dom if r['baseline']=='D']),indent=2))
