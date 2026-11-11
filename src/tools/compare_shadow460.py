from pathlib import Path
import json
import numpy as np
R=Path('/home/scv/shared/phase460_20261106')
data={}
for arm in ['A','B']:
 d=(Path('/home/scv/shared/phase459_20261106') if arm=='A' else R)/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow']
 data[arm]=json.loads((d/f'INDEPENDENT_{step}.json').read_text())
assert [r['index'] for r in data['A']['per_image']]==[r['index'] for r in data['B']['per_image']]
groups=sorted({r['group'] for r in data['A']['per_image']})
cm=np.array([[sum((np.array(r['confusion']) for r in data[a]['per_image'] if r['group']==g),start=np.zeros((3,3),np.int64)) for a in ['A','B']] for g in groups])
rng=np.random.default_rng(45902);draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
m=np.einsum('bn,nkij->bkij',draws,cm);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);delta=iou[:,1]-iou[:,0]
result=dict(B_minus_A_IoU_delta=(np.array(data['B']['result']['IoU'])-np.array(data['A']['result']['IoU'])).tolist(),CI95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),ROI_count=len(groups),bootstrap_draws=2000,selection='Each arm best trained shadow on same535 validation set; exploratory')
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(result,indent=2));np.savez_compressed(R/'MATCHED_BOOTSTRAP.npz',groups=np.array(groups),confusions=cm,draws=draws,IoU_delta=delta)
# Fixed-step comparison avoids differences in the selected training duration.
h={a:{r['step']:r for r in json.loads(((Path('/home/scv/shared/phase459_20261106') if a=='A' else R)/a/'network_run/HISTORY.json').read_text())} for a in ['A','B']}
result['matched_steps']=[dict(step=s,A=h['A'][s]['IoU'],B=h['B'][s]['IoU'],B_minus_A=(np.array(h['B'][s]['IoU'])-np.array(h['A'][s]['IoU'])).tolist()) for s in sorted(h['A'])]
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

off=json.loads((R/'B/network_run'/f"OFF_INDEPENDENT_{data['B']['result']['step']}.json").read_text())
cms=np.array([[sum((np.array(r['confusion']) for r in source['per_image'] if r['group']==g),start=np.zeros((3,3),np.int64)) for source in [off,data['B']]] for g in groups])
m=np.einsum('bn,nkij->bkij',draws,cms);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);diff=iou[:,1]-iou[:,0]
result['B_enabled_minus_disabled']=dict(IoU_delta=(np.array(data['B']['result']['IoU'])-np.array(off['result']['IoU'])).tolist(),CI95=np.quantile(diff,[.025,.975],axis=0).T.tolist())
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['B_enabled_minus_disabled']))

reference_off=json.loads((R/'B/network_run'/f"REFERENCE_OFF_INDEPENDENT_{data['B']['result']['step']}.json").read_text())
cms=np.array([[sum((np.array(r['confusion']) for r in source['per_image'] if r['group']==g),start=np.zeros((3,3),np.int64)) for source in [reference_off,data['B']]] for g in groups])
m=np.einsum('bn,nkij->bkij',draws,cms);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);diff=iou[:,1]-iou[:,0]
result['B_reference_on_minus_off']=dict(IoU_delta=(np.array(data['B']['result']['IoU'])-np.array(reference_off['result']['IoU'])).tolist(),CI95=np.quantile(diff,[.025,.975],axis=0).T.tolist())
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['B_reference_on_minus_off']))
