from collect_phase241 import ssh
from pathlib import Path
b=ssh('''from pathlib import Path
import numpy as np,json
r=Path('/home/scv/shared/phase448_20261106/network_run');a=np.load('/home/scv/shared/phase445_20261106/network_run/BOOTSTRAP_6000.npz');b=np.load(r/'BOOTSTRAP_6000.npz')
assert np.array_equal(a['groups'],b['groups'])
paired=np.stack([a['confusions'][:,1],b['confusions'][:,1]],axis=1);draws=b['draws'];m=np.einsum('bn,nkij->bkij',draws,paired);d=np.diagonal(m,axis1=-2,axis2=-1);iou=100*d/(m.sum(-1)+m.sum(-2)-d);diff=iou[:,1]-iou[:,0]
v=dict(comparison='448 minus445, fixed step6000, paired107ROIs,2000draws',IoU_delta_CI95=np.quantile(diff,[.025,.975],axis=0).T.tolist(),mIoU_delta_CI95=np.quantile(diff.mean(1),[.025,.975]).tolist())
(r/'PAIRED_VS445.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
''')
r=Path(__file__).resolve().parents[2]/'outputs/phase448/phase448_20261106/network_run';(r/'PAIRED_VS445.json').write_bytes(b);print(b.decode())
