from pathlib import Path
import json,numpy as np
R=Path('/home/scv/shared/phase444_20261103');OLD=Path('/home/scv/shared/phase429_20261102/catalogue_best/baseline')
rows=json.loads(Path('/home/scv/shared/phase443_20261103/INPUTS.json').read_text());targets={r['rgb_path']:r for r in json.loads(Path('/home/scv/shared/phase400_20261031/INPUT_LOCK.json').read_text())['target_rows'] if r['split']=='development_val'}
out=[]
for row in rows:
 if row['domain']!='B':continue
 item=targets[row['image']]
 with np.load(OLD/(item['product']+'.npz')) as old,np.load(R/'baseline429/predictions'/f"{row['index']:03d}.npz") as new:
  count=int((old['pred']!=new['original']).sum());out.append(dict(index=row['index'],different=count,pixels=old['pred'].size))
different=sum(v['different'] for v in out);pixels=sum(v['pixels'] for v in out)
assert different/pixels<1e-5
result=dict(passed=True,different=different,pixels=pixels,fraction=different/pixels,per_image=out)
(R/'BASELINE_ENDPOINT_PIXEL_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
