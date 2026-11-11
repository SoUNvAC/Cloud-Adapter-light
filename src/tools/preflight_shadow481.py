from pathlib import Path
import hashlib,json
R=Path('/home/scv/shared/phase481_20261108');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'shadow_redistribution481.py')==sha(Path('/home/scv/shared/phase480_20261108/shadow_redistribution480.py'))
sources={}
for arm in ['C','A','B']:
 d=Path('/home/scv/shared/phase477_20261107/A/network_run') if arm=='C' else Path('/home/scv/shared/phase480_20261108')/arm/'network_run'
 f=json.loads((d/'FINAL_RESULTS.json').read_text());assert f['passed']
 names=['FINAL_RESULTS.json','COMPLETE.json','LOCK.json','INPUT_TRACE.jsonl']+[f'INDEPENDENT_{s}.json' for s in set(f['choices'].values())]
 if arm!='C':names+=['ABLATION_OFF.json',f"OFF_INDEPENDENT_{f['choices']['best_shadow']}.json"]
 sources[arm]=dict(directory=str(d),sha256={n:sha(d/n) for n in names})
for seed in [45202,45203]:assert not (R/f'seed_{seed}').exists()
result=dict(passed=True,seed0=45201,module_byte_identical=True,module_sha=sha(R/'shadow_redistribution481.py'),sources=sources)
(R/'PREFLIGHT.json').write_text(json.dumps(result,indent=2));print('PREFLIGHT passed')
