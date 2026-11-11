from pathlib import Path
import hashlib,json
R=Path('/home/scv/shared/phase494_20261109');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'shadow_redistribution494.py')==sha(Path('/home/scv/shared/phase493_20261109/shadow_redistribution493.py'))
refs={}
for seed in [45201,45202,45203]:
 for method in ['S','N']:
  if method=='N' and seed!=45201:continue
  if seed==45201:d=Path('/home/scv/shared')/('phase493_20261109' if method=='N' else 'phase488_20261108')/'B/network_run'
  else:d=Path('/home/scv/shared/phase489_20261108')/f'seed_{seed}'/'B/network_run'
  f=json.loads((d/'FINAL_RESULTS.json').read_text());assert f['passed'] and f['target_label_supervision_percent']==0
  names=['FINAL_RESULTS.json','COMPLETE.json','LOCK.json','INPUT_TRACE.jsonl','branch_0.pth']+[f'INDEPENDENT_{s}.json' for s in set(f['choices'].values())]
  if method=='N':names += [f"OFF_INDEPENDENT_{f['choices']['best_shadow']}.json",f"REGION_OFF_INDEPENDENT_{f['choices']['best_shadow']}.json"]
  refs[f'{seed}/{method}']=dict(directory=str(d),sha256={n:sha(d/n) for n in names})
for seed in [45202,45203]:assert not (R/f'seed_{seed}').exists()
(R/'PREFLIGHT.json').write_text(json.dumps(dict(passed=True,module_byte_identical=True,refs=refs),indent=2));print('PREFLIGHT passed')
