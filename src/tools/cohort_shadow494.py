from pathlib import Path
import json,numpy as np
R=Path('/home/scv/shared/phase494_20261109');read=lambda p:json.loads(p.read_text());refs=read(R/'PREFLIGHT.json')['refs'];out=[]
for seed in [45201,45202,45203]:
 a=Path(refs[f'{seed}/S']['directory'])
 b=Path(refs[f'{seed}/N']['directory']) if seed==45201 else R/f'seed_{seed}'/'B/network_run'
 old=read(a/'INDEPENDENT_6000.json');new=read(b/'INDEPENDENT_6000.json')
 assert [(x['index'],x['group'],x['baseline_confusion']) for x in old['per_image']]==[(x['index'],x['group'],x['baseline_confusion']) for x in new['per_image']]
 def errors(v):
  c=np.array(v['result']['confusion']);d=np.diag(c);return c.sum(1)-d,c.sum(0)-d
 fn0,fp0=errors(old);fn1,fp1=errors(new)
 out.append(dict(seed=seed,FN_delta=(fn1-fn0).tolist(),FP_delta=(fp1-fp0).tolist()))
(R/'PAIR_COHORT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,paired_indices_groups_baseline_confusions_identical=True,comparison='493B/494 minus 488B/489 at fixed6000',errors=out),indent=2));print('PAIR_COHORT_ACCEPTANCE passed')
