"""All six subset trajectories share each input; fixed reference counts only."""
from phase237_common import *
import sys,traceback
def candidate(subset,seed,st,step=None,alpha=None):return dict(subset=subset,seed=seed,strategy=st,step=step,alpha=alpha,source=step is None)
def cid(v):return json.dumps({k:v[k] for k in ['subset','seed','strategy','step','alpha','source']},sort_keys=True)
def main():
 global budget
 stage=sys.argv[1];assert stage in ['val','test'];assert not (R/f'{stage.upper()}_STARTED.json').exists();check_bundle();save(f'{stage.upper()}_STARTED.json',{'epoch':time.time(),'PID':os.getpid()})
 if stage=='test':
  assert sha(R/'VAL_SELECTION_LOCK.json')==js(R/'VAL_SELECTION_SEAL.json')['SHA'];selection=js(R/'VAL_SELECTION_LOCK.json')['subset_selections'];assert len(selection)==18;save('TEST_ACCESS_STARTED.json',{'epoch':time.time(),'selection_SHA':sha(R/'VAL_SELECTION_LOCK.json')})
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 expected=js(R/'EXPECTED_SOURCE236.json')[stage];E=js(R/'EVAL_INPUTS.json');T=js(R/'SUBSET_INPUTS.json');trainpaths={str(Path(v[k]).resolve()) for rows in T.values() for v in rows for k in ['image_path','mask_path']};testpaths={str(Path(v[k]).resolve()) for v in E['test'] for k in ['image_path','mask_path']}
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=str(Path(os.fsdecode(args[0])).resolve())
   if p in trainpaths or (stage=='val' and p in testpaths) or any(x in p.lower() for x in ['l8_biome','cloudsen12_high_l1c']):
    if Path(p).suffix.lower() in ['.png','.npz','.jpg','.tif','.jp2']:raise RuntimeError('237 forbidden pixel access '+p)
 sys.addaudithook(audit)
 assert sha(SOURCE)==js(R/'LOCK.json')['Source_SHA256']
 for dep in js(R/'DEPENDENCY_LOCK.json').values():assert sha(dep['path'])==dep['SHA']
 budget=Budget('evaluate_'+stage);wrapper,_=build(budget);model=wrapper.segmentor.eval();model.requires_grad_(False);initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};frozen={n:t.detach().cpu().clone() for n,t in dict(list(model.named_parameters())+list(model.named_buffers())).items() if not n.startswith('decode_head.')}
 choices=[candidate('Source',None,'Source')]
 if stage=='val':
  for subset in ['SUBSET_1','SUBSET_2','SUBSET_3']:
   for seed in [21701,21702]:
    for st in ['E','W','P']:
     choices.append(candidate(subset,seed,st))
     for v in ([500,1000,2000,4000] if st=='E' else [.25,.5,.75,1]):choices.append(candidate(subset,seed,st,v if st=='E' else 4000,None if st=='E' else v))
 else:
  choices.extend({k:v[k] for k in ['subset','seed','strategy','step','alpha','source']} for v in selection)
  for subset in ['SUBSET_1','SUBSET_2','SUBSET_3']:
   for seed in [21701,21702]:choices.append(candidate(subset,seed,'endpoint',4000,1))
 def netkey(v):
  if v['source']:return 'Source'
  a=v['alpha'] if v['strategy']=='W' else 1
  return f'{v["subset"]}:{v["seed"]}:{v["step"]}:{a:g}'
 nets={'Source':initial};hashes={}
 for v in choices:
  if v['source']:continue
  p=R/'fits'/f'{v["subset"]}_{v["seed"]}'/f'head_{v["step"]}.pth';assert sha(p)==js(p.parent/f'SAVE_{v["step"]}.json')['head_SHA'];hashes[p.relative_to(R).as_posix()]=sha(p);endpoint=torch.load(p,map_location='cpu')['state_dict'];a=v['alpha'] if v['strategy']=='W' else 1
  nets[netkey(v)]=endpoint if a==1 else {n:(initial[n].double()+a*(endpoint[n].double()-initial[n].double())).to(initial[n].dtype) for n in initial}
 assert len(nets)<=(43 if stage=='val' else 19)
 totals={cid(v):np.zeros((3,3),np.int64) for v in choices};scenes={cid(v):{} for v in choices};rows=E[stage];sourceparity=[];records=(R/f'{stage}_PER_INPUT.jsonl').open('x',buffering=1)
 def store(v,pred,row,y):
  valid=y!=255;c=np.bincount((3*y[valid].astype(np.int64)+pred[valid]).ravel(),minlength=9).reshape(3,3);assert c.sum()==y.size and all(c.sum(1)==np.sum(y[None]==np.arange(3)[:,None,None],axis=(1,2)))
  total=cid(v);totals[total]+=c;scenes[total].setdefault(row['scene'],np.zeros((3,3),np.int64));scenes[total][row['scene']]+=c;h=hashlib.sha256(pred.astype(np.uint8).tobytes()).hexdigest()
  if v['source']:assert c.tolist()==expected[row['name']]['confusion'] and h==expected[row['name']]['prediction_SHA']
  records.write(json.dumps({**v,'split':stage,'name':row['name'],'scene':row['scene'],'confusion':c.tolist(),'prediction_SHA':h})+'\n')
 for index,row in enumerate(rows):
  assert sha(row['image_path'])==row['image_sha256'] and sha(row['mask_path'])==row['mask_sha256'];y=load_parent_target(Path(row['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8);assert set(np.unique(y))<={0,1,2};x=torch.from_numpy(load_rgb_image(Path(row['image_path']),512)).cuda();sourcep=None
  for nk,head in nets.items():
   model.decode_head.load_state_dict(head);budget.current={'stage':stage,'index':index,'name':row['name'],'network':nk};budget.count('evaluation_full_input')
   with torch.inference_mode():p=wrapper(x).float().clamp_min(0);p=p/p.sum(1,keepdim=True).clamp_min(1e-12)
   if nk=='Source':sourcep=p.clone()
   for v in choices:
    if nk!=netkey(v):continue
    value=(1-v['alpha'])*sourcep+v['alpha']*p if v['strategy']=='P' and not v['source'] else p;store(v,value.argmax(1)[0].cpu().numpy(),row,y)
   del p
  sourceparity.append(row['name']);del x,sourcep;budget.check()
 records.close()
 def score(c):
  c=np.asarray(c,np.float64);t=c.diagonal();u=c.sum(0)+c.sum(1)-t;return float(np.mean(t[u>0]/u[u>0])*100)
 table=[{**json.loads(k),'confusion':v.tolist(),'pooled_mIoU':score(v),'scenes':{g:c.tolist() for g,c in scenes[k].items()}} for k,v in totals.items()];save(stage.upper()+'_SUMMARY.json',table)
 if stage=='val':
  selections=[]
  for subset in ['SUBSET_1','SUBSET_2','SUBSET_3']:
   for seed in [21701,21702]:
    for st in ['E','W','P']:
     pool=[v for v in table if (v['subset'],v['seed'],v['strategy'])==(subset,seed,st)];assert len(pool)==5
     def order(v):return (-v['pooled_mIoU'],0,0) if v['source'] else (-v['pooled_mIoU'],1,v['step'] if st=='E' else v['alpha'])
     selections.append({k:v for k,v in min(pool,key=order).items() if k!='scenes'})
  refs=js(R/'FULL60_VAL_SELECTION.json');assert len(refs)==6;save('VAL_SELECTION_LOCK.json',{'subset_selections':selections,'full60_selections':refs,'all_subset_val_SHA':sha(R/'VAL_SUMMARY.json'),'full60_refs_SHA':sha(R/'FULL60_REFERENCES.json'),'heads_SHA':{p.relative_to(R).as_posix():sha(p) for p in (R/'fits').glob('*/head_*.pth')},'input_SHA':sha(R/'EVAL_INPUTS.json'),'code_SHA':{p.name:sha(p) for p in (R/'code').glob('*.py')}});save('VAL_SELECTION_SEAL.json',{'SHA':sha(R/'VAL_SELECTION_LOCK.json'),'epoch':time.time()})
 states=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,states[n].detach().cpu()) for n,t in frozen.items())
 save(stage.upper()+'_COMPLETE.json',{'stage':stage,'networks_per_input':len(nets),'inputs':len(rows),'per_input_SHA':sha(R/f'{stage}_PER_INPUT.jsonl'),'heads_SHA':hashes,'Source_exact236_all_inputs':sourceparity,'all_nonhead_Source_exact':True,'no_saved_scores':True});budget.finish('passed')
if __name__=='__main__':
 try:main()
 except BaseException:
  save(sys.argv[1].upper()+'_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise
