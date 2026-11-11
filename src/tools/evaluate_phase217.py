"""Fixed217 inference: four frozen endpoints, common216 Source cache, source-only locks."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,os,resource,sys,threading,time,traceback
ROOT=Path('/home/scv/shared/phase217_20261023');OUT=ROOT/'evaluation';SRC=Path('/home/scv/Cloud-Adapter-light/src');OLD=Path('/home/scv/shared/phase216_20261023')
SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';SH='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9';AS=[0,.25,.5,.75,1];SEEDS=[21701,21702]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def save(n,x):
 p=OUT/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str,allow_nan=False));t.replace(p)
def main():
 OUT.mkdir(exist_ok=False);assert (ROOT/'FOUR_ENDPOINTS_SEALED.json').exists();prior=js(ROOT/'BUDGET_STATE.json');assert prior['units']==40010
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SOURCE_TO_PARENT,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
 assert sha(SOURCE)==SH
 start=time.monotonic();counts=collections.Counter();stage='build';seed=None;arm=None;alpha=None;index=0;done=threading.Event();samples=[]
 ledger=(OUT/'CALL_LEDGER.jsonl').open('x',buffering=1)
 def status():save('RUN_STATUS.json',{'stage':stage,'seed':seed,'arm':arm,'alpha':alpha,'input':index,'counts':dict(counts),'new_units':sum(counts.values()),'cumulative_units':prior['units']+sum(counts.values()),'cumulative_GPU_seconds':prior['GPU_seconds']+time.monotonic()-start,'PID':os.getpid()})
 def count(k):
  assert prior['units']+sum(counts.values())<60000 and prior['GPU_seconds']+time.monotonic()-start<14400
  counts[k]+=1;ledger.write(json.dumps({'ordinal':prior['units']+sum(counts.values()),'kind':k,'stage':stage,'seed':seed,'arm':arm,'alpha':alpha,'input':index})+'\n')
 def guard():
  last=0
  from controller_phase217 import host
  while not done.wait(15):
   try:
    cpu=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0;reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    assert cpu<16*1024**3 and max(gpu,reserved)<20*1024**3 and size<4_000_000_000 and prior['GPU_seconds']+time.monotonic()-start<14400
    if time.monotonic()-last>180:host();last=time.monotonic()
    samples.append({'cpu_peak':cpu,'gpu_peak':gpu,'gpu_reserved_peak':reserved,'bytes':size});save('RESOURCE_SAMPLES.json',samples);status()
   except BaseException:
    (OUT/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(OUT/'EXIT_CODE').write_text('71\n');os._exit(71)
 def deny_source(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.path.abspath(os.fsdecode(args[0])).replace('\\','/')
   if 'cloudsen12_high_l1c/ann_dir/train/' in p or 'cloudsen12_high_l1c/ann_dir/test/' in p or 'cloudsen12_high_l1c/img_dir/test/' in p:raise RuntimeError('217 forbids CloudSEN train labels and official test access')
 sys.addaudithook(deny_source);threading.Thread(target=guard,daemon=True).start()
 try:
  support=js(ROOT/'PHASE216_EVAL_INPUT_LOCK.json');assert [len(support[p]) for p in ['source','val','test']]==[535,10,10]
  for pool,rows in support.items():
   for r in rows:assert sha(r['image_path'])==r['image_sha256'] and sha(r['mask_path'])==r['mask_sha256']
  save('INPUT_SUPPORT_LOCK.json',support)
  caches={};expected=js(ROOT/'SOURCE_CACHE_SHAS.json')
  # Load only source metrics now. Target caches remain unopened until selection is frozen.
  assert sha(OLD/'evaluation/Source__source.json')==expected['source'];caches['source']=js(OLD/'evaluation/Source__source.json')
  wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SOURCE),precision='fp32',active_block_indices=None));model=wrapper.segmentor.eval()
  for p in model.parameters():p.requires_grad=False
  all_t=dict(list(model.named_parameters())+list(model.named_buffers()));ck=torch.load(SOURCE,map_location='cpu')['state_dict'];assert all(torch.equal(t,all_t[n].detach().cpu()) for n,t in ck.items())
  initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};frozen={n:t.detach().cpu().clone() for n,t in all_t.items() if not n.startswith('decode_head.')};assert not dict(model.decode_head.named_buffers())
  (ROOT/'interpolated_heads').mkdir(exist_ok=False);heads={};heads_receipt={}
  for s in SEEDS:
   for a in ['C','D']:
    p=ROOT/f'seed{s}/heads/{a}_iter4000.pth';assert sha(p)==js(ROOT/f'seed{s}/{a}_TRAIN_COMPLETE.json')['head_SHA'];endpoint=torch.load(p,map_location='cpu')['state_dict'];assert set(endpoint)==set(initial)
    for f in AS[1:]:
     h={n:(endpoint[n].clone() if f==1 else (t.double()+f*(endpoint[n].double()-t.double())).to(t.dtype)) for n,t in initial.items()};assert all(torch.isfinite(t).all() for t in h.values());heads[s,a,f]=h
     if f!=1:
      p=ROOT/f'interpolated_heads/{s}_{a}_a{f}.pth';torch.save({'state_dict':h,'seed':s,'arm':a,'alpha':f,'Source_SHA':SH,'formula':'CPU float64 then original dtype'},p);heads_receipt[p.name]=sha(p)
  save('INTERPOLATED_HEAD_HASHES.json',heads_receipt)
  def sethead(h):
   model.decode_head.load_state_dict(h);model.eval();now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
  maps={'source':SOURCE_TO_PARENT,'val':SPARCS_TO_PARENT,'test':SPARCS_TO_PARENT}
  def one(r,pool,kind):
   x=torch.from_numpy(load_rgb_image(Path(r['image_path']),512)).cuda();count(kind)
   with torch.inference_mode():
    z=wrapper(x).float().clamp_min(0);pred=(z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
   y=load_parent_target(Path(r['mask_path']),maps[pool],pred.shape);valid=y!=255;return np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3),pred
  parity=[]
  def reproduce(pool):
   nonlocal stage,index,seed,arm,alpha
   stage='Source_parity_'+pool;index=0;seed=None;arm='Source';alpha=0;sethead(initial);r=support[pool][0];cm,pred=one(r,pool,'Source_parity_full_segmentor');old=next(x for x in caches[pool]['per_input'] if x['name']==r['name']);assert cm.tolist()==old['confusion'],('Source parity failure',pool,r['name'])
   parity.append({'pool':pool,'input':r['name'],'confusion':cm.tolist(),'matches216':True});save('SOURCE_PARITY.json',parity)
  reproduce('source');save('Source__source.json',caches['source'])
  def evaluate(s,a,f,pool):
   nonlocal stage,seed,arm,alpha,index
   stage=pool;seed=s;arm=a;alpha=f;sethead(heads[s,a,f]);gs=collections.defaultdict(lambda:np.zeros((3,3),np.int64));total=np.zeros((3,3),np.int64);per=[];ph=hashlib.sha256()
   for index,r in enumerate(support[pool],1):
    c,p=one(r,pool,'candidate_full_segmentor');total+=c;gs[r['scene']]+=c;ph.update(p.tobytes());per.append({'name':r['name'],'scene':r['scene'],'confusion':c.tolist()})
   result={'seed':s,'arm':a,'alpha':f,'pool':pool,'confusion':total.tolist(),'per_group':{g:c.tolist() for g,c in sorted(gs.items())},'per_input':per,'prediction_SHA':ph.hexdigest()};save(f'{s}_{a}_a{f}__{pool}.json',result);status();return result
  for s in SEEDS:
   for a in ['C','D']:
    for f in AS[1:]:evaluate(s,a,f,'source')
  def score(c):
   c=np.asarray(c,np.float64);d=np.diag(c);u=c.sum(0)+c.sum(1)-d;return float(100*np.nanmean(np.divide(d,u,out=np.full(3,np.nan),where=u>0)))
  base=caches['source']['per_group'];assert len(base)==107;sm={};selections=[]
  for s in SEEDS:
   for a in ['C','D']:
    for f in AS:
     r=caches['source'] if f==0 else js(OUT/f'{s}_{a}_a{f}__source.json');g=r['per_group'];assert set(g)==set(base) and all(np.array_equal(np.asarray(g[k]).sum(1),np.asarray(base[k]).sum(1)) for k in g)
     ls=sorted((score(base[k])-score(g[k]) for k in g),reverse=True);sm[f'{s}/{a}/{f}']={'pool_loss_pp':score(caches['source']['confusion'])-score(r['confusion']),'T11_pp':float(np.mean(ls[:11])),'worst_ROI_loss_pp':ls[0],'ROI_loss_gt5':sum(x>5 for x in ls)}
    for b in [1,3,5]:selections.append({'seed':s,'arm':a,'budget_pp':b,'alpha':max(f for f in AS if sm[f'{s}/{a}/{f}']['pool_loss_pp']<=b and sm[f'{s}/{a}/{f}']['T11_pp']<=5)})
  save('SOURCE_SELECTION_LOCK.json',{'epoch':time.time(),'rows':selections,'source_metrics':sm,'target_metrics_opened_before_lock':False,'rule':'same216 largest alpha pool_loss<=B AND signed T11<=5, no epsilon','budgets':[1,3,5]})
  for pool in ['val','test']:
   path=OLD/f'evaluation/Source__{pool}.json';assert sha(path)==expected[pool];caches[pool]=js(path);reproduce(pool);save(f'Source__{pool}.json',caches[pool])
  for s in SEEDS:
   for a in ['C','D']:
    for f in AS[1:]:
     for pool in ['val','test']:evaluate(s,a,f,pool)
  assert counts=={'Source_parity_full_segmentor':3,'candidate_full_segmentor':8880};done.set()
  save('COMPLETE.json',{'total_new_units':sum(counts.values()),'cumulative_units':prior['units']+sum(counts.values()),'counts':dict(counts),'GPU_seconds':time.monotonic()-start,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'Source_SHA_unchanged':sha(SOURCE)==SH,'all_common_Source_checks':parity,'evaluation_flags':{'deterministic_algorithms':False,'cudnn_deterministic':False,'cudnn_benchmark':False,'precision':'FP32','argmax':'historical sum-normalized semantic scores'},'source_selection_SHA':sha(OUT/'SOURCE_SELECTION_LOCK.json')});stage='complete';status();(OUT/'EXIT_CODE').write_text('0\n')
 except BaseException:
  done.set();(OUT/'FAILURE.txt').write_text(traceback.format_exc());status();save('FAILURE_COST.json',{'new_units':sum(counts.values()),'cumulative_units':prior['units']+sum(counts.values()),'GPU_seconds':time.monotonic()-start});(OUT/'EXIT_CODE').write_text('1\n');raise
 finally:done.set();ledger.close()
if __name__=='__main__':main()
