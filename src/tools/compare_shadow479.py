from pathlib import Path
import json,numpy as np
R=Path('/home/scv/shared/phase479_20261108');P=Path('/home/scv/shared/phase477_20261107/A/network_run')
read=lambda p:json.loads(p.read_text());data={};pairs=[]
for s in [1000,3000,6000]:data[f'477_{s}']=read(P/f'INDEPENDENT_{s}.json')
data['477best']=read(P/f"INDEPENDENT_{read(P/'FINAL_RESULTS.json')['choices']['best_shadow']}.json")
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json')
 for s in [1000,3000,6000]:
  name=f'479{arm}_{s}';data[name]=read(d/f'INDEPENDENT_{s}.json');pairs.append((name,f'477_{s}'))
 name=f'479{arm}best';data[name]=read(d/f"INDEPENDENT_{final['choices']['best_shadow']}.json");pairs.append((name,'477best'))
for arm in ['A','B']:
 d=R/arm/'network_run';step=read(d/'FINAL_RESULTS.json')['choices']['best_shadow']
 data[f'479{arm}off']=read(d/f'OFF_INDEPENDENT_{step}.json');pairs.append((f'479{arm}best',f'479{arm}off'))
for step in [1000,3000,6000]:pairs.append((f'479B_{step}',f'479A_{step}'))
groups=sorted({v['group'] for v in data['477best']['per_image']});assert len(groups)==107
draws=np.random.default_rng(47902).multinomial(107,np.full(107,1/107),size=2000)
def compare(a,b):
 assert [(v['index'],v['group']) for v in data[a]['per_image']]==[(v['index'],v['group']) for v in data[b]['per_image']]
 cm=np.array([[sum((np.array(v['confusion']) for v in data[n]['per_image'] if v['group']==g),start=np.zeros((3,3),np.int64)) for n in [b,a]] for g in groups])
 m=np.einsum('bn,nkij->bkij',draws,cm);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);delta=iou[:,1]-iou[:,0]
 totals=cm.sum(0);td=np.diagonal(totals,axis1=-2,axis2=-1);fn=totals.sum(-1)-td;fp=totals.sum(-2)-td
 return dict(IoU_delta=(np.array(data[a]['result']['IoU'])-np.array(data[b]['result']['IoU'])).tolist(),CI95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),steps=[data[a]['result']['step'],data[b]['result']['step']],TP_delta=(td[1]-td[0]).tolist(),FN_delta=(fn[1]-fn[0]).tolist(),FP_delta=(fp[1]-fp[0]).tolist())
out={a+'_minus_'+b:compare(a,b) for a,b in pairs};out.update(ROI_count=107,bootstrap_draws=2000,selection='Single seed, fixed-step and validation-selected best comparisons')
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(out,indent=2));np.savez_compressed(R/'MATCHED_BOOTSTRAP.npz',groups=np.array(groups),draws=draws);print('MATCHED_COMPARISON complete')
