"""Phase213: fixed source-budget alpha selection, then target-test only."""
from pathlib import Path
import collections, csv, hashlib, json, os, sys, time, traceback

REPO=Path('/home/scv/Cloud-Adapter-light'); SRC=REPO/'src'; BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02'); ROOT=Path('/home/scv/shared/phase213_20261021_run03'); OUT=ROOT/'evaluation'
SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'; SOURCE_SHA='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
HEAD_SHA={'l8':'cad66c102421230d1749b8c2e3bcfed36a47591a380484ecb8b000abf6114c31','sparcs':'98a2c1844463b8dbfaaca87354e84006a72059b16e3199636c5a445f45956c8e'}
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 t=Path(str(OUT/n)+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str));t.replace(OUT/n)
def metrics(c):
 import numpy as np
 x=np.asarray(c,dtype=np.float64);tp=np.diag(x);u=x.sum(0)+x.sum(1)-tp;i=np.divide(tp,u,out=np.full(3,np.nan),where=u>0);return {'mIoU':100*float(np.nanmean(i)),'Surface_IoU':100*float(i[0]) if np.isfinite(i[0]) else None,'Cloud_IoU':100*float(i[1]) if np.isfinite(i[1]) else None,'Shadow_IoU':100*float(i[2]) if np.isfinite(i[2]) else None,'Shadow_recall':float(x[2,2]/x[2].sum()) if x[2].sum() else None,'nonShadow_to_Shadow_FPR':float(x[:2,2].sum()/x[:2].sum()) if x[:2].sum() else None,'Shadow_to_Surface':float(x[2,0]/x[2].sum()) if x[2].sum() else None,'Shadow_to_Cloud':float(x[2,1]/x[2].sum()) if x[2].sum() else None,'valid_pixels':int(x.sum()),'shadow_reference_pixels':int(x[2].sum()),'nonshadow_reference_pixels':int(x[:2].sum())}
def main():
 os.chdir(SRC);sys.path.insert(0,str(SRC));sys.path.insert(0,str(SRC/'tools'));os.environ['XFORMERS_DISABLED']='1'
 import numpy as np, torch
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import manifest_records,load_parent_target,L8_TO_PARENT,SPARCS_TO_PARENT
 lock=json.loads((ROOT/'ALPHA_SELECTION_LOCK.json').read_text());assert lock['test_read_before_lock'] is False and lock['new_target_calls_planned']==10885
 assert sha(SOURCE)==SOURCE_SHA
 OUT.mkdir(exist_ok=False);(OUT/'heads').mkdir(); save('RUN_STATUS.json',{'status':'running','stage':'bootstrap','calls':0,'expected_calls':10885})
 manifests={'l8':SRC/'work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv','sparcs':SRC/'work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv'}
 rows={t:manifest_records(p,'target_test') for t,p in manifests.items()};assert len(rows['l8'])==2167 and len(rows['sparcs'])==10
 maps={'l8':L8_TO_PARENT,'sparcs':SPARCS_TO_PARENT};calls=0;clock=time.monotonic();ledger=(OUT/'CALL_LEDGER.jsonl').open('x',buffering=1)
 save('TEST_INPUT_LOCK.json',{'manifest_sha256':{t:sha(p) for t,p in manifests.items()},'counts':{t:{'inputs':len(rows[t]),'scenes':len({x['scene'] for x in rows[t]})} for t in rows},'paths_stat_only_verified':True,'test_labels_opened_after_alpha_lock':True,'mapping':{'l8':L8_TO_PARENT.tolist(),'sparcs':SPARCS_TO_PARENT.tolist()},'candidates':[0,.25,.5,.75,1],'source_val_selection_lock_sha256':sha(ROOT/'ALPHA_SELECTION_LOCK.json')})
 wrapper,cfg=build_wrapper(__import__('argparse').Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SOURCE),precision='fp32',active_block_indices=None));wrapper.eval()
 named=dict(list(wrapper.segmentor.named_parameters())+list(wrapper.segmentor.named_buffers()));source_head={n:t.detach().cpu().clone() for n,t in named.items() if n.startswith('decode_head.')};frozen={n:t.detach().cpu().clone() for n,t in named.items() if not n.startswith('decode_head.')}
 heads={}
 for t in ['l8','sparcs']:
  th=torch.load(BASE/t/('best_mIoU_iter_3500.pth' if t=='l8' else 'best_mIoU_iter_3000.pth'),map_location='cpu')['state_dict'];assert sha(BASE/t/('best_mIoU_iter_3500.pth' if t=='l8' else 'best_mIoU_iter_3000.pth'))==HEAD_SHA[t];assert set(th)==set(source_head)
  heads[t]={}
  for a in [0,.25,.5,.75,1]:
   h={n:(source_head[n].clone() if a==0 else th[n].clone() if a==1 else (source_head[n].double()+a*(th[n].double()-source_head[n].double())).to(source_head[n].dtype)) for n in source_head}
   assert all(torch.isfinite(v).all() for v in h.values());heads[t][a]=h;torch.save({'state_dict':h,'alpha':a,'target':t,'Source_SHA':SOURCE_SHA,'target_SHA':HEAD_SHA[t]},OUT/'heads'/f'{t}_a{str(a).replace(".","p")}.pth')
 def loadhead(h):
  state=wrapper.segmentor.state_dict();state.update(h);wrapper.segmentor.load_state_dict(state,strict=True);now=dict(list(wrapper.segmentor.named_parameters())+list(wrapper.segmentor.named_buffers()));assert all(torch.equal(now[n].detach().cpu(),v) for n,v in frozen.items())
 def evaluate(t,a):
  nonlocal calls
  loadhead(heads[t][a]);total=np.zeros((3,3),np.int64);groups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));per=[];ph=hashlib.sha256()
  with torch.inference_mode():
   for i,r in enumerate(rows[t]):
    assert calls<12000 and time.monotonic()-clock<10800
    p=Path(r['image_path']);p=p if p.is_absolute() else SRC/p; m=Path(r['mask_path']);m=m if m.is_absolute() else SRC/m
    x=torch.from_numpy(load_rgb_image(p,512)).cuda();calls+=1;ledger.write(json.dumps({'call':calls,'target':t,'alpha':a,'input':i})+'\n');s=wrapper(x).float().clamp_min(0);s=s/s.sum(1,keepdim=True).clamp_min(1e-12);pred=s.argmax(1)[0].cpu().numpy().astype(np.uint8);ph.update(pred.tobytes());y=load_parent_target(m,maps[t],pred.shape);valid=y!=255;cm=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3);total+=cm;groups[r['scene']]+=cm;per.append({'name':r['name'],'scene':r['scene'],'confusion':cm.tolist()})
  res={'target':t,'alpha':a,'confusion':total.tolist(),'per_scene':{k:v.tolist() for k,v in sorted(groups.items())},'per_input':per,'prediction_sha256':ph.hexdigest(),'inputs':len(rows[t]),'scenes':len(groups),'metrics':metrics(total),'group_mean_mIoU':float(np.mean([metrics(v)['mIoU'] for v in groups.values()])),'group_max_FPR':max((metrics(v)['nonShadow_to_Shadow_FPR'] for v in groups.values() if metrics(v)['nonShadow_to_Shadow_FPR'] is not None),default=None)};save(f'{t}_a{str(a).replace(".","p")}.json',res);return res
 results=[]
 for t in ['l8','sparcs']:
  for a in [0,.25,.5,.75,1]:results.append(evaluate(t,a))
 save('RUN_STATUS.json',{'status':'complete','calls':calls}); save('RUN_COMPLETE.json',{'status':'target_test_complete','calls':calls,'elapsed_seconds':time.monotonic()-clock,'expected_calls':10885,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'results':len(results),'test_labels_read_after_lock':True});(OUT/'EXIT_CODE').write_text('0\n');ledger.close()
if __name__=='__main__':main()
