"""All H interpolation points; source-only selection then full target evaluation."""
from pathlib import Path
import argparse,copy,collections,hashlib,json,os,resource,sys,threading,time,traceback
R=Path('/home/scv/shared/phase224_20261024');O=R/'evaluation';S=Path('/home/scv/Cloud-Adapter-light/src');AA=[0,.25,.5,.75,1]
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False,default=str));t.replace(p)
def main():
 assert not O.exists();O.mkdir();assert (R/'FOUR_ENDPOINTS_SEALED.json').exists();lock=js(R/'EXPERIMENT_LOCK.json');prior=js(R/'BUDGET_STATE.json');assert prior['units']==64016;os.chdir(S);sys.path[:0]=[str(S),str(S/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 import torch.nn.functional as F
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SOURCE_TO_PARENT,L8_TO_PARENT,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False;torch.set_grad_enabled(False)
 start=time.monotonic();counts=collections.Counter();stage='build';job=None;alpha=None;pool=None;index=0;done=threading.Event();ledger=(O/'CALL_LEDGER.jsonl').open('x',buffering=1);samples=[]
 def status():save('RUN_STATUS.json',{'stage':stage,'job':job,'alpha':alpha,'pool':pool,'input':index,'counts':dict(counts),'new_units':sum(counts.values()),'cumulative_units':prior['units']+sum(counts.values()),'GPU_seconds':time.monotonic()-start,'PID':os.getpid()})
 def count(k):
  assert prior['units']+sum(counts.values())<120000 and prior['GPU_seconds']+time.monotonic()-start<21600;counts[k]+=1;ledger.write(json.dumps({'ordinal':prior['units']+sum(counts.values()),'kind':k,'job':job,'alpha':alpha,'pool':pool,'input':index})+'\n')
 def guard():
  from controller_phase224 import host
  last=0
  while not done.wait(15):
   try:
    v={'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,'GPU_reserved_peak':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))};assert v['CPU_peak_bytes']<16*1024**3 and max(v['GPU_allocated_peak'],v['GPU_reserved_peak'])<20*1024**3 and v['bytes']<4000000000 and prior['GPU_seconds']+time.monotonic()-start<21600
    if time.monotonic()-last>180:v['host_volumes']=host();last=time.monotonic()
    samples.append(v);save('RESOURCE_SAMPLES.json',samples);status()
   except BaseException:(O/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(O/'EXIT_CODE').write_text('71\n');os._exit(71)
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.path.abspath(os.fsdecode(args[0])).replace('\\','/')
   if any(v in p for v in ['cloudsen12_high_l1c/ann_dir/train/','cloudsen12_high_l1c/ann_dir/test/','cloudsen12_high_l1c/img_dir/test/']):raise RuntimeError('224 eval forbidden Source reference/test input '+p)
 sys.addaudithook(audit);threading.Thread(target=guard,daemon=True).start()
 try:
  source=S/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9';support=js(R/'INPUT_SUPPORT_LOCK.json');wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(source),precision='fp32',active_block_indices=None));model=wrapper.segmentor.eval()
  for p in model.parameters():p.requires_grad=False
  full=dict(list(model.named_parameters())+list(model.named_buffers()));ck=torch.load(source,map_location='cpu')['state_dict'];assert all(torch.equal(t,full[n].detach().cpu()) for n,t in ck.items());initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};frozen={n:t.detach().cpu().clone() for n,t in full.items() if not n.startswith('decode_head.')};assert not dict(model.decode_head.named_buffers());teacher=copy.deepcopy(model.decode_head).eval();partner=copy.deepcopy(model.decode_head).eval();heads={};dheads={};interp={};(R/'interpolated_heads').mkdir()
  for key,meta in lock['jobs'].items():
   p=R/key/'heads/H_iter4000.pth';assert sha(p)==js(R/key/'H_TRAIN_COMPLETE.json')['head_SHA'];endpoint=torch.load(p,map_location='cpu')['state_dict'];assert set(endpoint)==set(initial);oldp=Path(meta['historical_D_path']);assert sha(oldp)==meta['historical_D_SHA'];dheads[key]=torch.load(oldp,map_location='cpu')['state_dict']
   for f in AA[1:]:
    h={n:(endpoint[n].clone() if f==1 else (v.double()+f*(endpoint[n].double()-v.double())).to(v.dtype)) for n,v in initial.items()};assert all(torch.isfinite(v).all() for v in h.values());heads[key,f]=h
    if f!=1:
     p=R/f'interpolated_heads/{key}_H_a{f}.pth';torch.save({'state_dict':h,'alpha':f,'arm':'H','target':meta['target'],'seed':meta['seed'],'Source_SHA':sha(source),'formula':'CPU float64 then source dtype'},p);interp[p.name]=sha(p)
  save('INTERPOLATED_HEAD_HASHES.json',interp);captured={};hook=model.decode_head.register_forward_pre_hook(lambda m,args:captured.update(features=args[0]));teacher_cache={};oldD={};oldDH={};oldcm=collections.defaultdict(lambda:np.zeros((3,3),np.int64));oldgroups=collections.defaultdict(lambda:collections.defaultdict(lambda:np.zeros((3,3),np.int64)));gates=[];parity=[];jointfile=(O/'JOINT_COUNTS.jsonl').open('x',buffering=1)
  for key,meta in lock['jobs'].items():
   oldD[key]=js(R/f"historical/{meta['target']}/evaluation/{meta['seed']}_D_a1__source.json");oldDH[key]=hashlib.sha256()
  caches={'source':js(R/'historical/SPARCS/evaluation/Source__source.json')};cache_source_by={v['name']:v for v in caches['source']['per_input']}
  def sethead(h):
   model.decode_head.load_state_dict(h);model.eval();now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
  def pred(z):
   z=z.float().clamp_min(0);return (z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
  def rawsha(a):return hashlib.sha256(a.tobytes()).hexdigest()
  def semantic(h,features,k):
   count(k);c,m=h(features,None);return torch.einsum('bqc,bqhw->bchw',c[-1].float().softmax(-1)[...,:-1],F.interpolate(m[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid())
  def one(row,k,paired=False):
   x=torch.from_numpy(load_rgb_image(Path(row['image_path']),512)).cuda();count(k)
   with torch.inference_mode():
    hp=pred(wrapper(x));features=captured.pop('features')
    if paired:
     partner.load_state_dict(dheads[job]);partner.eval();ds=semantic(partner,features,'historical_D_head');dp=pred(ds)
     if row['name'] not in teacher_cache:tp=pred(semantic(teacher,features,'diagnostic_teacher_head'));teacher_cache[row['name']]=tp
     else:tp=teacher_cache[row['name']]
     if index==1:
      sethead(dheads[job]);count('shared_D_gate_full');official=wrapper(x);captured.clear();assert torch.equal(official,ds);sethead(heads[job,1]);gates.append({'job':job,'name':row['name'],'semantic_bitwise_equal':True})
   mapping=SOURCE_TO_PARENT if pool=='source' else L8_TO_PARENT if pool.startswith('L8') else SPARCS_TO_PARENT;y=load_parent_target(Path(row['mask_path']),mapping,(512,512));valid=y!=255;c=np.bincount(3*y[valid].astype(np.int64)+hp[valid],minlength=9).reshape(3,3)
   if paired:
    dc=np.bincount(3*y[valid].astype(np.int64)+dp[valid],minlength=9).reshape(3,3);tc=np.bincount(3*y[valid].astype(np.int64)+tp[valid],minlength=9).reshape(3,3);old=next(v for v in oldD[job]['per_input'] if v['name']==row['name']);assert dc.tolist()==old['confusion'] and tc.tolist()==cache_source_by[row['name']]['confusion']
    if 'prediction_SHA' in old:assert rawsha(dp)==old['prediction_SHA']
    oldDH[job].update(dp.tobytes());oldcm[job]+=dc;oldgroups[job][row['scene']]+=dc;j=np.bincount(27*y[valid].astype(np.int64)+9*tp[valid]+3*dp[valid]+hp[valid],minlength=81).reshape(3,3,3,3);meta=lock['jobs'][job];jointfile.write(json.dumps({'target':meta['target'],'seed':meta['seed'],'job':job,'name':row['name'],'ROI':row['scene'],'valid_pixels':int(valid.sum()),'J':j.tolist(),'teacher_prediction_SHA':rawsha(tp),'D_prediction_SHA':rawsha(dp),'H_prediction_SHA':rawsha(hp)})+'\n')
   return c,hp
  def reproduce(key):
   nonlocal stage,pool,job,alpha,index
   stage='Source_parity';pool=key;job=None;alpha=0;index=0;sethead(initial);row=support[key][0];c,p=one(row,'Source_parity_full');old=next(v for v in caches[key]['per_input'] if v['name']==row['name']);assert c.tolist()==old['confusion'];parity.append({'pool':key,'name':row['name'],'confusion':c.tolist(),'exact':True});save('SOURCE_PARITY.json',parity);save(f'Source__{key}.json',caches[key])
  reproduce('source')
  def evaluate(key,f,which):
   nonlocal stage,pool,job,alpha,index
   meta=lock['jobs'][key];job=key;alpha=f;pool='source' if which=='source' else meta['target']+'_'+which;stage='evaluate_'+which;sethead(heads[key,f]);groups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));total=np.zeros((3,3),np.int64);per=[];ph=hashlib.sha256()
   for index,row in enumerate(support[pool],1):
    c,p=one(row,'candidate_full_segmentor',paired=which=='source' and f==1);groups[row['scene']]+=c;total+=c;ph.update(p.tobytes());per.append({'name':row['name'],'scene':row['scene'],'confusion':c.tolist(),'prediction_SHA':rawsha(p)})
   save(f"{meta['target']}_{meta['seed']}_H_a{f}__{which}.json",{'target':meta['target'],'seed':meta['seed'],'arm':'H','alpha':f,'pool':which,'confusion':total.tolist(),'per_group':{g:c.tolist() for g,c in sorted(groups.items())},'per_input':per,'prediction_SHA':ph.hexdigest()});status()
  for key in lock['jobs']:
   for f in AA[1:]:evaluate(key,f,'source')
  jointfile.close();reproduction={}
  for key,e in oldD.items():
   assert oldcm[key].tolist()==e['confusion'] and {g:c.tolist() for g,c in oldgroups[key].items()}==e['per_group'] and oldDH[key].hexdigest()==e['prediction_SHA'];reproduction[key]={'all535_inputs_and107_ROIs_exact':True,'prediction_SHA':oldDH[key].hexdigest(),'expected_SHA':e['prediction_SHA']}
  save('HISTORICAL_D_REPRODUCTION.json',reproduction);save('SHARED_PATH_GATES.json',gates)
  def score(c):
   c=np.asarray(c,np.float64);d=np.diag(c);u=c.sum(0)+c.sum(1)-d;return float(100*np.nanmean(np.divide(d,u,out=np.full(3,np.nan),where=u>0)))
  sm={};choices=[];base=caches['source'];bg=base['per_group']
  for key,meta in lock['jobs'].items():
   for f in AA:
    q=base if f==0 else js(O/f"{meta['target']}_{meta['seed']}_H_a{f}__source.json");ls=sorted([score(bg[g])-score(q['per_group'][g]) for g in bg],reverse=True);sm[f'{key}/{f}']={'pool_loss_pp':score(base['confusion'])-score(q['confusion']),'T11_pp':float(np.mean(ls[:11])),'worst_ROI_loss_pp':ls[0],'ROI_loss_gt5':sum(v>5 for v in ls)}
   for b in [1,3,5]:choices.append({'target':meta['target'],'seed':meta['seed'],'arm':'H','budget_pp':b,'alpha':max(f for f in AA if sm[f'{key}/{f}']['pool_loss_pp']<=b and sm[f'{key}/{f}']['T11_pp']<=5)})
  save('SOURCE_SELECTION_LOCK.json',{'epoch':time.time(),'rows':choices,'source_metrics':sm,'target_metrics_opened_before_lock':False,'rule':'largest alpha pool_loss<=B AND signed T11<=5; no epsilon','budgets':[1,3,5]})
  for t in ['SPARCS','L8']:
   for which in ['val','test']:
    k=t+'_'+which;caches[k]=js(R/f'historical/{t}/evaluation/Source__{which}.json');reproduce(k)
  for key in lock['jobs']:
   for f in AA[1:]:
    for p in ['val','test']:evaluate(key,f,p)
  assert counts=={'Source_parity_full':5,'candidate_full_segmentor':33760,'historical_D_head':2140,'diagnostic_teacher_head':535,'shared_D_gate_full':4};done.set();stage='complete';save('COMPLETE.json',{'total_units':sum(counts.values()),'counts':dict(counts),'GPU_seconds':time.monotonic()-start,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),'source_selection_SHA':sha(O/'SOURCE_SELECTION_LOCK.json'),'all_Source_parity':parity,'historical_D_reproduction':reproduction,'evaluation_flags':{'deterministic_algorithms':False,'cudnn_deterministic':False,'cudnn_benchmark':False,'precision':'FP32'}});status();(O/'EXIT_CODE').write_text('0\n')
 except BaseException:done.set();(O/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(counts),'total_units':sum(counts.values()),'GPU_seconds':time.monotonic()-start});status();(O/'EXIT_CODE').write_text('1\n');raise
 finally:done.set();ledger.close()
if __name__=='__main__':main()
