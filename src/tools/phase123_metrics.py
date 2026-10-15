"""Same accepted operators, synchronized group bootstrap and within-group seed means."""
import types,json,csv
from pathlib import Path
import numpy as np
import phase122_metrics as accepted
from phase120_metrics import table
source=Path(accepted.__file__).read_text(encoding='utf-8').replace('122015','123015')
engine=types.ModuleType('phase123_accepted_metric_engine');exec(compile(source,accepted.__file__,'exec'),engine.__dict__)
METRICS=('AP','recall','precision','FPR','Surface_FPR','Cloud_FPR')
def average(a,b):return (a+b)/2 if a is not None and b is not None else None
def evaluate(groups,pool,thresholds,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);summaries=[];rows=[];pairs=[];states={}
    for arm in ('C122','C123'):
        folder=out/arm
        g={gid:dict(products=v['products'],truth=v['truth'],R=v['R'],C=v[arm]) for gid,v in groups.items()}
        ss,cc=engine.evaluate(g,pool,{'R':thresholds['R'],'C':thresholds[arm]},folder)
        with (folder/(pool+'_per_group.csv')).open(encoding='utf-8',newline='') as f:rr=list(csv.DictReader(f))
        for v in rr:
            if arm=='C123' and v['setting'].startswith('R_'):continue
            v['setting']=v['setting'].replace('C100',arm)
            for key in METRICS:v[key]=None if v[key]=='' else float(v[key])
            rows.append(v);states[v['group_id'],v['setting']]=v
        for v in ss:
            if arm=='C123' and v['setting'].startswith('R_'):continue
            v['setting']=v['setting'].replace('C100',arm);summaries.append(v)
        for v in cc:
            if arm=='C123' and v['comparison']=='R_cal-R_05':continue
            v['comparison']=v['comparison'].replace('C100',arm);v['primary']=False;pairs.append(v)
    meanrows=[];deltas=[]
    for gid in sorted(groups):
        a,b,r=states[gid,'C122_cal'],states[gid,'C123_cal'],states[gid,'R_cal']
        row=dict(pool=pool,group_id=gid,products=groups[gid]['products'],setting='C_two_seed_metric_mean_cal',not_score_ensemble=True)
        delta=dict(pool=pool,group_id=gid,comparison='C_two_seed_metric_mean_cal-R_cal')
        for key in METRICS:
            row[key]=average(a[key],b[key]);delta[key+'_delta_pp']=(row[key]-r[key])*100 if row[key] is not None and r[key] is not None else None
        meanrows.append(row);deltas.append(delta)
    main=dict(pool=pool,comparison='C_two_seed_metric_mean_cal-R_cal',primary=pool=='Catalogue',seed=123015,resamples=10000,
        unit='related group; seed metrics averaged inside each group, not independent seeds and not score ensemble')
    mean=dict(pool=pool,setting='C_two_seed_metric_mean_cal',not_score_ensemble=True,all_groups=len(groups))
    for key in METRICS:
        main[key]=engine.interval([v[key+'_delta_pp'] for v in deltas])
        mean[key]=engine.interval([v[key]*100 if v[key] is not None else None for v in meanrows])
        valid=[(v[key+'_delta_pp'],v['group_id']) for v in deltas if v[key+'_delta_pp'] is not None]
        main[key+'_minimum_delta_pp']=min(valid)[0] if valid else None;main[key+'_maximum_delta_pp']=max(valid)[0] if valid else None
    mean['maximum_group_FPR_pp']=max(v['FPR'] for v in meanrows if v['FPR'] is not None)*100
    mean['over_1pct_FPR_groups']=sum(v['FPR']>.01 for v in meanrows if v['FPR'] is not None)
    mean['pooled_AP_metric_mean_pp']=average(*[next(v['pooled_AP_pp'] for v in summaries if v['setting']==arm+'_cal') for arm in ('C122','C123')])
    table(out/(pool+'_six_workpoints.csv'),summaries);table(out/(pool+'_per_group.csv'),rows)
    table(out/(pool+'_two_seed_group_means.csv'),meanrows);table(out/(pool+'_primary_group_differences.csv'),deltas)
    with (out/(pool+'_comparisons.json')).open('x',encoding='utf-8') as f:json.dump(dict(primary=main,seed_and_operating_comparisons=pairs,two_seed_metric_mean_summary=mean),f,indent=2,allow_nan=False)
    return summaries,main
