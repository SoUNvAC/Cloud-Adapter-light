from pathlib import Path
import hashlib,json
R=Path('/home/scv/shared/phase483_20261108');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'shadow_redistribution483.py')==sha(Path('/home/scv/shared/phase482_20261108/shadow_redistribution482.py'))
refs={}
for seed in [45201,45202,45203]:
 for method in ['C','S','N']:
  if method=='N' and seed!=45201:continue
  if seed==45201:
   d={'C':Path('/home/scv/shared/phase477_20261107/A/network_run'),'S':Path('/home/scv/shared/phase480_20261108/B/network_run'),'N':Path('/home/scv/shared/phase482_20261108/B/network_run')}[method]
  else:d=Path('/home/scv/shared/phase481_20261108')/f'seed_{seed}'/('C' if method=='C' else 'B')/'network_run'
  f=json.loads((d/'FINAL_RESULTS.json').read_text());assert f['passed'] and f['target_label_supervision_percent']==0
  names=['FINAL_RESULTS.json','COMPLETE.json','LOCK.json','INPUT_TRACE.jsonl','branch_0.pth']+[f'INDEPENDENT_{s}.json' for s in set(f['choices'].values())]
  if method=='N':names += ['ABLATION_OFF.json',f"OFF_INDEPENDENT_{f['choices']['best_shadow']}.json"]
  refs[f'{seed}/{method}']=dict(directory=str(d),sha256={n:sha(d/n) for n in names})
for seed in [45202,45203]:assert not (R/f'seed_{seed}').exists()
(R/'PREFLIGHT.json').write_text(json.dumps(dict(passed=True,module_byte_identical=True,refs=refs),indent=2));print('PREFLIGHT passed')
