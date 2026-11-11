"""Fixed val candidates followed by sealed test selections, no score archive."""
from phase236_common import *
import copy,sys,traceback

def main():
 global budget
 stage=sys.argv[1];assert stage in ['val','test'];assert not (R/f'{stage.upper()}_STARTED.json').exists();check_bundle()
 save(f'{stage.upper()}_STARTED.json',{'epoch':time.time(),'PID':os.getpid()})
 if stage=='test':assert sha(R/'VAL_SELECTION_LOCK.json')==js(R/'VAL_SELECTION_SEAL.json')['SHA'];selection=js(R/'VAL_SELECTION_LOCK.json')['selections'];save('TEST_ACCESS_STARTED.json',{'epoch':time.time(),'selection_SHA':sha(R/'VAL_SELECTION_LOCK.json')})
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch,rasterio
 from rasterio.windows import Window
 from PIL import Image
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 assert sha(SOURCE)==js(R/'LOCK.json')['Source_SHA256']
 budget=Budget('evaluate_recovery03_'+stage);wrapper,_=build(budget);model=wrapper.segmentor.eval();model.requires_grad_(False);initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()}
 frozen={n:t.detach().cpu().clone() for n,t in dict(list(model.named_parameters())+list(model.named_buffers())).items() if not n.startswith('decode_head.')};E=js(R/'EVAL_INPUTS.json');T=js(R/'TRAIN_INPUTS.json');plan=js(R/'NATIVE_PLAN.json');valrefs={v['name']:v for v in js(R/'L8_VAL_NATIVE_REFERENCES.json')};expected={(v['split'],v['name']):v for v in js(R/'EXPECTED_NATIVE234.json')};handles={};native_receipts=[]
 trainpaths={str(Path(v[k]).resolve()) for rows in T.values() for v in rows for k in ['image_path','mask_path']};testpaths={str(Path(v[k]).resolve()) for pools in E.values() for v in pools['test'] for k in ['image_path','mask_path']}
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=str(Path(os.fsdecode(args[0])).resolve())
   if p in trainpaths or (stage=='val' and p in testpaths) or ('cloudsen12_high_l1c' in p and Path(p).suffix.lower() in ['.png','.npz','.tif']):raise RuntimeError('236 evaluation access denied '+p)
 sys.addaudithook(audit)
 def label(v,domain):
  if domain=='SPARCS':return load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8)
  if stage=='val':
   r=valrefs[v['name']];assert sha(r['file'])==r['sha256']
   with np.load(r['file'],allow_pickle=False) as z:a=z['native'];valid=z['valid'];truth=z['truth']
  else:
   p=plan['test'][v['scene']]
   if v['scene'] not in handles:assert sha(p['path'])==p['SHA'];handles[v['scene']]=rasterio.open(p['path'])
   a=handles[v['scene']].read(1,window=Window(int(v['y']),int(v['x']),512,512),boundless=True,fill_value=0);valid=a!=0;truth=np.zeros(a.shape,np.uint8);truth[a==64]=2;truth[(a==192)|(a==255)]=1;truth[~valid]=255
  lut=np.zeros(256,np.uint8);lut[64]=1;lut[192]=2;lut[255]=3;assert np.array_equal(lut[a],np.asarray(Image.open(v['mask_path'])))
  ex=expected[stage,v['name']];assert hashlib.sha256(a.tobytes()).hexdigest()==ex['native_array_SHA'];assert hashlib.sha256(valid.tobytes()).hexdigest()==ex['valid_array_SHA']
  native_receipts.append({'stage':stage,'name':v['name'],'native_SHA':ex['native_array_SHA'],'valid_SHA':ex['valid_array_SHA'],'support':[int((truth==c).sum()) for c in range(3)],'Fill':int((truth==255).sum()),'old_PNG_exact':True})
  return truth
 def path(domain,seed,lr,step):return R/'fits'/f'{domain}_{seed}_{lr:.0e}'/f'head_{step}.pth'
 def key(seed,lr,step,alpha=1):return f'{seed}:{lr:.0e}:{step}:{alpha}'
 def miou(cm):
  c=np.asarray(cm,np.float64);i=c.diagonal();u=c.sum(0)+c.sum(1)-i;return float(np.mean(i[u>0]/u[u>0])*100)
 def candidate(seed,strategy,lr=None,step=None,alpha=None):return {'seed':seed,'strategy':strategy,'lr':lr,'step':step,'alpha':alpha,'source':lr is None}
 def netkey(c):return 'Source' if c['source'] else key(c['seed'],c['lr'],c['step'],c['alpha'] if c['strategy']=='W' else 1)
 summary={};allchoices=[]
 for domain in ['L8','SPARCS']:
  candidates=[]
  if stage=='val':
   for seed in [21701,21702]:
    for strategy in ['E','W','P']:
     candidates.append(candidate(seed,strategy))
     for lr in [1e-4,3e-5]:
      for value in ([500,1000,2000,4000] if strategy=='E' else [.25,.5,.75,1]):candidates.append(candidate(seed,strategy,lr,value if strategy=='E' else 4000,None if strategy=='E' else value))
  else:
   for s in selection:
    if s['domain']==domain:candidates.append({k:v for k,v in s.items() if k in ['seed','strategy','lr','step','alpha','source']})
   for seed in [21701,21702]:
    for lr in [1e-4,3e-5]:candidates.append(candidate(seed,'endpoint',lr,4000,1))
   candidates.append(candidate(None,'Source'))
  def cid(c):return json.dumps(c,sort_keys=True)
  nets={'Source':initial};headhash={}
  # At most29 small heads, no image/feature score cache.
  for c in candidates:
   if c['source']:continue
   p=path(domain,c['seed'],c['lr'],c['step']);assert sha(p)==js(p.parent/f'SAVE_{c["step"]}.json')['head_SHA'];headhash[str(p.relative_to(R))]=sha(p)
   ep=torch.load(p,map_location='cpu')['state_dict'];a=c['alpha'] if c['strategy']=='W' else 1
   d=ep if a==1 else {n:(initial[n].double()+a*(ep[n].double()-initial[n].double())).to(initial[n].dtype) for n in initial}
   nets[netkey(c)]=d
   if c['strategy']=='P':nets[key(c['seed'],c['lr'],4000,1)]=ep
  assert len(nets)<=(29 if stage=='val' else 9),(domain,stage,len(nets))
  totals={cid(c):np.zeros((3,3),np.int64) for c in candidates};scenes={cid(c):{} for c in candidates};record=(R/f'{domain}_{stage}_PER_INPUT.jsonl').open('x',buffering=1)
  def store(c,pred,v,y):
   ident=cid(c);valid=y!=255;cm=np.bincount((3*y[valid].astype(np.int64)+pred[valid]).ravel(),minlength=9).reshape(3,3);totals[ident]+=cm;scenes[ident].setdefault(v['scene'],np.zeros((3,3),np.int64));scenes[ident][v['scene']]+=cm
   record.write(json.dumps({'domain':domain,'split':stage,'name':v['name'],'scene':v['scene'],'usgs_shadows':v.get('usgs_shadows'),**c,'confusion':cm.tolist(),'prediction_SHA':hashlib.sha256(pred.astype(np.uint8).tobytes()).hexdigest()})+'\n')
  rows=E[domain][stage]
  for index,v in enumerate(rows):
   assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256'];y=label(v,domain);x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda();sourcep=None
   for nk,head in nets.items():
    model.decode_head.load_state_dict(head);budget.current={'domain':domain,'split':stage,'input':index,'name':v['name'],'network':nk};budget.count('evaluation_full_input')
    with torch.inference_mode():
     p=wrapper(x).float().clamp_min(0);p=p/p.sum(1,keepdim=True).clamp_min(1e-12)
    if nk=='Source':sourcep=p.clone()
    pred=p.argmax(1)[0].cpu().numpy()
    for c in candidates:
     if c['strategy']=='P' and not c['source']:
      if nk==netkey(c):
       assert sourcep is not None;mixed=(1-c['alpha'])*sourcep+c['alpha']*p;store(c,mixed.argmax(1)[0].cpu().numpy(),v,y)
     elif nk==netkey(c):store(c,pred,v,y)
    del p,pred
   del x,sourcep
   if index%100==0:budget.check()
  record.close();table=[{'domain':domain,**json.loads(k),'confusion':v.tolist(),'pooled_mIoU':miou(v),'scenes':{g:c.tolist() for g,c in scenes[k].items()}} for k,v in totals.items()]
  save(f'{domain}_{stage}_SUMMARY.json',table);summary[domain]={'networks_per_input':len(nets),'inputs':len(rows),'heads':headhash,'per_input_SHA':sha(R/f'{domain}_{stage}_PER_INPUT.jsonl')}
  if stage=='val':
   for seed in [21701,21702]:
    for strategy in ['E','W','P']:
     pool=[t for t in table if t['seed']==seed and t['strategy']==strategy]
     def order(c):
      if c['source']:return (-c['pooled_mIoU'],0,0,0)
      return (-c['pooled_mIoU'],1,c['step'] if strategy=='E' else c['alpha'],c['lr'])
     chosen=min(pool,key=order);allchoices.append({k:v for k,v in chosen.items() if k!='scenes'})
  states=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,states[n].detach().cpu()) for n,t in frozen.items())
 if stage=='val':
  assert len(allchoices)==12;save('VAL_SELECTION_LOCK.json',{'selections':allchoices,'val_tables':{d:sha(R/f'{d}_val_SUMMARY.json') for d in summary},'model_SHA':{p.relative_to(R).as_posix():sha(p) for p in (R/'fits').glob('*/head_*.pth')},'input_SHA':sha(R/'EVAL_INPUTS.json'),'code_SHA':{p.name:sha(p) for p in (R/'code').glob('*.py')},'rule':js(R/'LOCK.json')['selection']});save('VAL_SELECTION_SEAL.json',{'SHA':sha(R/'VAL_SELECTION_LOCK.json'),'epoch':time.time()})
 save(f'{stage.upper()}_NATIVE_ACCEPTANCE.json',native_receipts);save(f'{stage.upper()}_COMPLETE.json',{'stage':stage,'pools':summary,'no_saved_scores':True,'all_nonhead_states_Source':True});budget.finish('passed')
 for h in handles.values():h.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  save(sys.argv[1].upper()+'_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise
