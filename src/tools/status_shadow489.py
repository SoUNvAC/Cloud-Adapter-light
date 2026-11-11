from pathlib import Path
import json
R=Path('/home/scv/shared/phase489_20261108');read=lambda p:json.loads(p.read_text())
for seed in [45202,45203]:
 d=R/f'seed_{seed}'/'B/network_run';out=dict(seed=seed)
 if (d/'FAILURE.json').exists():raise RuntimeError((d/'FAILURE.json').read_text())
 for n in ['STATUS','COMPLETE','FINAL_RESULTS','ABLATION_OFF']:
  if (d/(n+'.json')).exists():out[n]=read(d/(n+'.json')) if n=='STATUS' else True
 out['done']=(d.parent/'DONE').exists()
 if (d/'INPUT_TRACE.jsonl').exists():
  old=Path('/home/scv/shared/phase483_20261108')/f'seed_{seed}'/'B/network_run/INPUT_TRACE.jsonl';old={x['step']:x for x in map(json.loads,old.read_text().splitlines())}
  trace=list(map(json.loads,(d/'INPUT_TRACE.jsonl').read_text().splitlines()));assert all(v==old[v['step']] for v in trace);out['matched_recorded_steps']=[v['step'] for v in trace]
 e=R/f'seed_{seed}_EXIT_CODE'
 if e.exists():assert e.read_text().strip()=='0',(seed,e.read_text())
 print(json.dumps(out))
