from pathlib import Path
import json,numpy as np
R=Path('/home/scv/shared/phase486_20261108');out={}
for arm in ['A','B']:
 v=json.loads((R/arm/'network_run/INDEPENDENT_6000.json').read_text());d=np.array([x['consensus_direction'] for x in v['per_image']])
 assert d.shape==(535,3) and np.isfinite(d).all() and np.max(np.abs(np.linalg.norm(d,axis=1)-1))<1e-5
 angle=np.degrees(np.arccos(np.clip(d.sum(1)/np.sqrt(3),-1,1)))
 if arm=='B':assert np.max(angle)<.05
 out[arm]=dict(images=535,mean_direction=d.mean(0).tolist(),std_direction=d.std(0).tolist(),angle_to_equal_RGB_degrees_quantiles=np.quantile(angle,[0,.25,.5,.75,1]).tolist())
(R/'DIRECTION_DIAGNOSTIC.json').write_text(json.dumps(dict(passed=True,arms=out),indent=2));print(json.dumps(out))
