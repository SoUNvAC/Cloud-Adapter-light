from pathlib import Path
import json,numpy as np
R=Path('/home/scv/shared/phase492_20261109');P=Path('/home/scv/shared/phase488_20261108/B/network_run')
read=lambda p:json.loads(p.read_text());data={};pairs=[]
for s in [1000,3000,6000]:data[f'488B_{s}']=read(P/f'INDEPENDENT_{s}.json')
data['488Bbest']=read(P/f"INDEPENDENT_{read(P/'FINAL_RESULTS.json')['choices']['best_shadow']}.json")
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json')
 for s in [1000,3000,6000]:
  name=f'492{arm}_{s}';data[name]=read(d/f'INDEPENDENT_{s}.json');pairs.append((name,f'488B_{s}'))
 step=final['choices']['best_shadow'];name=f'492{arm}best';data[name]=read(d/f'INDEPENDENT_{step}.json');pairs.append((name,'488Bbest'))
 data[f'492{arm}off']=read(d/f'OFF_INDEPENDENT_{step}.json');pairs.append((name,f'492{arm}off'))
 data[f'492{arm}regionoff']=read(d/f'REGION_OFF_INDEPENDENT_{step}.json');pairs.append((name,f'492{arm}regionoff'))
pairs.append(('492A_6000','492B_6000'))
groups=sorted({v['group'] for v in data['488Bbest']['per_image']});assert len(groups)==107
draws=np.random.default_rng(49202).multinomial(107,np.full(107,1/107),size=2000)
for name,v in data.items():
 assert len(v['per_image'])==535 and v['result']['state_unchanged']
 cm=sum((np.array(z['confusion']) for z in v['per_image']),start=np.zeros((3,3),np.int64));diag=np.diag(cm);iou=100*diag/(cm.sum(0)+cm.sum(1)-diag)
 assert np.array_equal(cm,v['result']['confusion']) and np.max(np.abs(iou-v['result']['IoU']))<1e-9
def compare(a,b):
 assert [(v['index'],v['group']) for v in data[a]['per_image']]==[(v['index'],v['group']) for v in data[b]['per_image']]
 cm=np.array([[sum((np.array(v['confusion']) for v in data[n]['per_image'] if v['group']==g),start=np.zeros((3,3),np.int64)) for n in [b,a]] for g in groups])
 m=np.einsum('bn,nkij->bkij',draws,cm);diag=np.diagonal(m,axis1=-2,axis2=-1);iou=100*diag/(m.sum(-1)+m.sum(-2)-diag);delta=iou[:,1]-iou[:,0]
 totals=cm.sum(0);td=np.diagonal(totals,axis1=-2,axis2=-1);fn=totals.sum(-1)-td;fp=totals.sum(-2)-td
 return dict(IoU_delta=(np.array(data[a]['result']['IoU'])-np.array(data[b]['result']['IoU'])).tolist(),CI95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),steps=[data[a]['result']['step'],data[b]['result']['step']],FN_delta=(fn[1]-fn[0]).tolist(),FP_delta=(fp[1]-fp[0]).tolist())
out={a+'_minus_'+b:compare(a,b) for a,b in pairs};out.update(ROI_count=107,bootstrap_draws=2000,selection='Single seed, matched steps primary; validation-selected best secondary')
old=read(P/'BOUNDARY_DIAG.json')['totals']['488']
out['boundary_best_vs_488B']={arm:{k:(np.array(read(R/arm/'network_run/BOUNDARY_DIAG.json')['totals']['492'][k])-old[k]).tolist() for k in ['FN','FP']} for arm in ['A','B']}
(R/'MATCHED_COMPARISON.json').write_text(json.dumps(out,indent=2));np.savez_compressed(R/'MATCHED_BOOTSTRAP.npz',groups=np.array(groups),draws=draws);print('MATCHED_COMPARISON passed')
