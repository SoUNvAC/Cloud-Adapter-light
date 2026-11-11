from pathlib import Path
import json
import numpy as np
R=Path('/home/scv/shared/phase477_20261107');D=R/'A/network_run';P=Path('/home/scv/shared/phase476_20261107')
read=lambda p:json.loads(p.read_text());final=read(D/'FINAL_RESULTS.json');best=final['choices']['best_shadow']
data={'477best':read(D/f'INDEPENDENT_{best}.json')}
for s in [1000,3000,6000]:data[f'477_{s}']=read(D/f'INDEPENDENT_{s}.json')
for arm in ['A','B']:
 d=P/arm/'network_run';step=read(d/'FINAL_RESULTS.json')['choices']['best_shadow']
 data['476'+arm]=read(d/f'INDEPENDENT_{step}.json');data['476'+arm+'_off']=read(d/f'OFF_INDEPENDENT_{step}.json')
groups=sorted({v['group'] for v in data['477best']['per_image']});assert len(groups)==107
rng=np.random.default_rng(47702);draws=rng.multinomial(107,np.full(107,1/107),size=2000)
def compare(a,b):
 assert [v['index'] for v in data[a]['per_image']]==[v['index'] for v in data[b]['per_image']]
 cm=np.array([[sum((np.array(v['confusion']) for v in data[n]['per_image'] if v['group']==g),start=np.zeros((3,3),np.int64)) for n in [b,a]] for g in groups])
 m=np.einsum('bn,nkij->bkij',draws,cm);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);delta=iou[:,1]-iou[:,0]
 return dict(IoU_delta=(np.array(data[a]['result']['IoU'])-np.array(data[b]['result']['IoU'])).tolist(),CI95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),steps=[data[a]['result']['step'],data[b]['result']['step']])
pairs=[('477best','476A'),('477best','476B'),('477best','476B_off'),('477_1000','476A'),('477_1000','476A_off'),('477_3000','476B'),('477_3000','476B_off')]
out={a+'_minus_'+b:compare(a,b) for a,b in pairs}
h={n:{r['step']:r for r in read(p/'HISTORY.json')} for n,p in [('477',D),('476A',P/'A/network_run'),('476B',P/'B/network_run')]}
out['matched_training_steps']=[dict(step=s,**{n:h[n][s]['IoU'] for n in h}) for s in sorted(h['477'])]
out.update(ROI_count=107,bootstrap_draws=2000,selection='Single-seed validation-selected best and fixed-step comparisons')
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(out,indent=2));np.savez_compressed(R/'MATCHED_BOOTSTRAP.npz',groups=np.array(groups),draws=draws)
print(json.dumps(out))
