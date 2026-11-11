from collect_phase241 import ssh
from pathlib import Path
b=ssh('''from pathlib import Path
import numpy as np,json
r=Path('/home/scv/shared/phase450_20261106/network_run');a=np.load('/home/scv/shared/phase449_20261106/network_run/BOOTSTRAP_6000.npz');selected=json.loads((r/'FINAL_RESULTS.json').read_text())['choices']['best_shadow'];b=np.load(r/f'BOOTSTRAP_{selected}.npz')
assert np.array_equal(a['groups'],b['groups'])
paired=np.stack([a['confusions'][:,1],b['confusions'][:,1]],axis=1);draws=b['draws'];m=np.einsum('bn,nkij->bkij',draws,paired);d=np.diagonal(m,axis1=-2,axis2=-1);iou=100*d/(m.sum(-1)+m.sum(-2)-d);diff=iou[:,1]-iou[:,0]
v=dict(comparison='450 minus449, selected models, paired107ROIs,2000draws',IoU_delta_CI95=np.quantile(diff,[.025,.975],axis=0).T.tolist(),mIoU_delta_CI95=np.quantile(diff.mean(1),[.025,.975]).tolist())
(r/'PAIRED_VS449.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
''')
r=Path(__file__).resolve().parents[2]/'outputs/phase450/phase450_20261106/network_run';(r/'PAIRED_VS449.json').write_bytes(b);print(b.decode())
