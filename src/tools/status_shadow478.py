from pathlib import Path
import json
R=Path('/home/scv/shared/phase478_20261108');P=Path('/home/scv/shared/phase477_20261107/A/network_run')
read=lambda p:json.loads(p.read_text())
old={x['step']:x for x in map(json.loads,(P/'INPUT_TRACE.jsonl').read_text().splitlines())}
for arm in ['A','B']:
 d=R/arm/'network_run';out={'arm':arm}
 if (d/'FAILURE.json').exists():raise RuntimeError((d/'FAILURE.json').read_text())
 for name in ['STATUS','COMPLETE','FINAL_RESULTS']:
  p=d/(name+'.json')
  if p.exists():out[name]=read(p) if name=='STATUS' else True
 if (d/'INPUT_TRACE.jsonl').exists():
  trace=[json.loads(x) for x in (d/'INPUT_TRACE.jsonl').read_text().splitlines()]
  assert all(x==old[x['step']] for x in trace)
  out['matched_batch_steps']=[x['step'] for x in trace]
 if (d/'HISTORY.json').exists():out['validated_steps']=[x['step'] for x in read(d/'HISTORY.json')]
 if (d/'LOCK.json').exists():out['trainable_parameters']=read(d/'LOCK.json')['task_parameters']
 print(json.dumps(out))
