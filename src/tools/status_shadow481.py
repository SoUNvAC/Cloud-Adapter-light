from pathlib import Path
import json
R=Path('/home/scv/shared/phase481_20261108');read=lambda p:json.loads(p.read_text())
for seed in [45202,45203]:
 traces={}
 for arm in ['C','A','B']:
  d=R/f'seed_{seed}'/arm/'network_run';out=dict(seed=seed,arm=arm)
  if not d.exists():out['stage']='queued'
  else:
   if (d/'FAILURE.json').exists():raise RuntimeError((d/'FAILURE.json').read_text())
   for name in ['STATUS','COMPLETE','FINAL_RESULTS','ABLATION_OFF']:
    if (d/(name+'.json')).exists():out[name]=read(d/(name+'.json')) if name=='STATUS' else True
   out['done']=(d.parent/'DONE').exists()
   if (d/'INPUT_TRACE.jsonl').exists():traces[arm]={v['step']:v for v in map(json.loads,(d/'INPUT_TRACE.jsonl').read_text().splitlines())}
  print(json.dumps(out))
 if len(traces)>1:
  shared=set.intersection(*(set(v) for v in traces.values()))
  assert all(all(v[s]==next(iter(traces.values()))[s] for v in traces.values()) for s in shared)
  print(json.dumps(dict(seed=seed,matched_recorded_steps=sorted(shared))))
for arm in ['C','A','B']:
 p=R/f'{arm}_EXIT_CODE'
 if p.exists():assert p.read_text().strip()=='0',(arm,p.read_text())
