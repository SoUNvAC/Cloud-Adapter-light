"""Fixed Q90 PL/CB: one teacher per image, one unique endpoint per arm, no target truth."""
import gc,random,traceback
import numpy as np
import torch
from torch.nn import functional as F
from phase317_common import *
from phase317_images import image,pseudo
import execute_phase302 as base
import worker_phase304 as legacy
STAGE,POOL,ARM=sys.argv[1:4];D=ROOT/POOL/ARM
CONTEXT='startup'
def context(s):
 global CONTEXT
 CONTEXT=s
def call(kind):
 p=ROOT/'CALLS.jsonl';count=sum(1 for _ in p.open()) if p.exists() else 0;assert count<11000
 append(p,dict(stage=STAGE,pool=POOL,arm=ARM,context=CONTEXT,kind=kind,epoch=time.time(),PID=os.getpid()))
def seeded():random.seed(30201);np.random.seed(30201);torch.manual_seed(30201);torch.cuda.manual_seed_all(30201)
def startup():
 assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python' and torch.__version__=='2.1.0+cu121'
 import mmcv,mmseg
 assert mmcv.__version__=='2.1.0' and mmseg.__version__=='1.2.2'
 resource_check();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
 from mmseg.utils import register_all_modules
 register_all_modules();sys.path.insert(0,str(R/'src'));import cloud_adapter.models
 for n,h in load(ROOT/'GPU_CODE_LOCK.json').items():assert sha(R/n)==h,n
 base.ROOT=D;base.count=call;base.seeded=seeded
 masks={str(Path(r['mask_path']).resolve()) for pool in ['SPARCS','L8'] for split in ['fit','val'] for r in inputs(pool)[split]}
 allowed_weights={str(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'),str(OLD/'initial_lora.pth'),str(D/'final_lora.pth')}
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   p=os.fsdecode(args[0]);low=p.replace('\\','/').lower()
   assert p not in masks and not any(s in low for s in ['/ann_dir/','_mask.png','masks.zip','alt_masks','classification_tags','ground_truth','pixbox','phase305','phase306','phase307']),p
   if low.endswith(('.pth','.pt')):assert p in allowed_weights or p.startswith(str(D)+'/'),p
 sys.addaudithook(guard)
def fresh(lora):seeded();return base.construct(lora)
def counts(p):return [int((p==c).sum()) for c in range(3)]
def losses(s,p,arm):
 target=torch.from_numpy(p.astype(np.int64))[None].cuda();pix=F.cross_entropy(s,target,ignore_index=255,reduction='none');means=[pix[target==c].mean() if bool((target==c).any()) else None for c in range(3)]
 assert any(v is not None for v in means)
 loss=pix[target!=255].mean() if arm=='PL' else torch.stack([v for v in means if v is not None]).mean()
 return loss,means
def toy():
 # Hand-calculated unequal-support example: classCE[1,3]/[10] ->CB6, PL14/3.
 ce=torch.tensor([1.,3.,10.]);y=torch.tensor([0,0,1]);cb=torch.stack([ce[y==c].mean() for c in [0,1]]).mean();assert float(cb)==6 and abs(float(ce.mean())-14/3)<1e-6
 same=torch.full((4,),float(np.log(3)));yy=torch.tensor([0,0,0,1]);assert abs(float(torch.stack([same[yy==c].mean() for c in [0,1]]).mean())-float(same.mean()))<1e-6
 assert float(ce[y==0].mean())==2.;assert not np.any(np.full((2,2),255,np.uint8)!=255)
 return dict(unequal_class_sizes_CB=6.,PL=14/3,equal_class_CE_invariant=True,singleclass_equals_PL=True,allignore_skips=True)
def gate_inputs():
 from mmengine.dataset import Compose
 model,state,names,cfg=fresh(False);checks=[];context('input_gate')
 for pool in ['SPARCS','L8']:
  row=sorted(inputs(pool)['val'],key=lambda r:r['name'])[0]
  pipe=Compose([dict(type='LoadImageFromFile'),dict(type='Resize',scale=(512,512)),dict(type='PackSegInputs')]);old=pipe(dict(img_path=row['image_path'],seg_fields=[]));a,_=image(pool,row)
  assert np.array_equal(a,old['inputs'].numpy().astype(np.float32));batch=model.data_preprocessor(dict(inputs=[old['inputs']],data_samples=[old['data_samples']]),training=False)
  current=base.preprocess(model,a,record=True);assert torch.equal(batch['inputs'],current['inputs'])
  with torch.no_grad():
   call('legacy_decode_predict');s=model.decode_head.predict(model.extract_feat(batch['inputs']),[x.metainfo for x in batch['data_samples']],model.test_cfg)
   z=base.forward(model,a,'new_input_Source');err=float((s-z).abs().max());assert err<=1e-6 and torch.equal(s.argmax(1),z.argmax(1))
  np.savez_compressed(ROOT/pool/'INPUT_GATE_SCORES.npz',scores=z[0].cpu().numpy(),input=a)
  checks.append(dict(pool=pool,name=row['name'],legacy_val_pipeline='LoadImage/Resize512/PackSegInputs with annotations omitted',input_equal=True,preprocessor_equal=True,max_score_error=err,pred_equal=True,output_space='resized512 decode_head scores; no return to1000px for thisnewfixedview protocol'))
 legacy.check_base(model,state,names);write(ROOT/'INPUT_ACCEPTANCE.json',dict(status='passed',rows=checks,CB_manual=toy()))
def teacher():
 assert POOL in ['SPARCS','L8'] and not (ROOT/POOL/'PSEUDO_LOCK.json').exists()
 model,state,names,_=fresh(False);rows=inputs(POOL)['fit'];out=[];context('teacher');td=ROOT/POOL/'teacher';td.mkdir()
 for row in rows:
  a,v=image(POOL,row)
  with torch.no_grad():s=base.forward(model,a,'teacher_fit')[0].cpu().numpy()
  total=s.sum(0);q=np.divide(s,total[None],out=np.zeros_like(s),where=total[None]>1e-12);p=np.where(v&(total>1e-12)&(q.max(0)>=.9),s.argmax(0),255).astype(np.uint8);path=td/(row['name']+'.npz');np.savez_compressed(path,scores=s,pred=s.argmax(0).astype(np.uint8),pseudo=p,valid=v)
  out.append(dict(name=row['name'],file=str(path),sha256=sha(path),pseudo_sha=ah(p),valid=int(v.sum()),classes=counts(p),accepted=int((p!=255).sum())))
 legacy.check_base(model,state,names);crop=[]
 for st in schedule(POOL):
  p=pseudo(POOL,rows[st['fit_index']],st);cc=counts(p);crop.append(dict(**st,mask_sha256=ah(p),Surface=cc[0],Cloud=cc[1],Shadow=cc[2],accepted=sum(cc)))
 table(ROOT/POOL/'CROP_IDENTITY.csv',crop);write(ROOT/POOL/'PSEUDO_LOCK.json',dict(status='sealed',rows=out,accepted=sum(r['accepted'] for r in out),classes=np.sum([r['classes'] for r in out],axis=0).tolist(),nonempty_products=sum(r['accepted']>0 for r in out),expected_updates=sum(r['accepted']>0 for r in crop),class_present_steps=[sum(r[c]>0 for r in crop) for c in C],schedule_sha=sha(ROOT/POOL/'SCHEDULE.json'),crop_sha=sha(ROOT/POOL/'CROP_IDENTITY.csv'),truth_read=False))
 print(POOL,'teacher',sum(r['accepted']>0 for r in crop),'updates',flush=True)
def train():
 assert (POOL,ARM) in ARMS and not (D/'initial_lora.pth').exists()
 assert all((ROOT/p/'PSEUDO_LOCK.json').exists() for p in ['SPARCS','L8']) and (ROOT/'INPUT_ACCEPTANCE.json').exists()
 rows=inputs(POOL)['fit'];steps=schedule(POOL);ctx=load(OLD/'PSEUDO_LOCK.json') if POOL=='CMC' else load(ROOT/POOL/'PSEUDO_LOCK.json')
 expected_updates=955 if POOL=='CMC' else ctx['expected_updates'];first=next((s for s in steps if (pseudo(POOL,rows[s['fit_index']],s)!=255).any()),steps[0]);a,_=image(POOL,rows[first['fit_index']],first);context('train_gate');source,state,_,_=fresh(False)
 with torch.no_grad():s=base.forward(source,a,'gate_Source_original',original=True)
 del source;gc.collect();torch.cuda.empty_cache();model,state,names,cfg=fresh(True);params=[p for n,p in model.named_parameters() if n in names];assert sum(p.numel() for p in params)==294912
 initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names};prior=torch.load(OLD/'initial_lora.pth',map_location='cpu');assert set(prior)==set(initial) and all(torch.equal(prior[n],v) for n,v in initial.items());assert all(torch.count_nonzero(v)==0 for n,v in initial.items() if n.endswith('.lora_b'))
 with torch.no_grad():z=base.forward(model,a,'gate_zero_lora');err=float((z-s).abs().max());assert err<=1e-6
 frozen=base.frozen_digest(model,names);assert frozen==load(OLD/'ENDPOINT_LOCK.json')['base_initial_final_sha256'];torch.save(initial,D/'initial_lora.pth')
 optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 p=pseudo(POOL,rows[first['fit_index']],first)
 if (p!=255).any():
  loss,_=losses(base.forward(model,a,'gate_gradient'),p,ARM);loss.backward();assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in params) and all(q.grad is None for n,q in model.named_parameters() if n not in names)
 with torch.no_grad():
  for n,q in model.named_parameters():
   if n in initial:q.copy_(initial[n])
 model.zero_grad(set_to_none=True);seeded();assert not optimizer.state and all(q.grad is None for q in params);legacy.check_base(model,state,names)
 cfg.dump(str(D/'resolved_config.py'));write(D/'MODEL_ACCEPTANCE.json',dict(status='passed',zero_lora_error=err,fresh_initial_equals304=True,initial_sha=sha(D/'initial_lora.pth'),initial_tensor_SHA={n:ah(v.numpy()) for n,v in initial.items()},B_zero=True,parameters=294912,only_lora_grad=True,initial_optimizer_grad_RNG_restored=True,frozen_digest=frozen,seed=30201,manual_CB=toy()))
 def probe(attempt):
  context('probe_'+str(attempt));probes=sorted(rows,key=lambda r:r.get('product',r.get('name')))[:4] if POOL=='CMC' else sorted(rows,key=lambda r:r['name'])[:1]
  for row in probes:
   a,v=image(POOL,row)
   if POOL=='CMC':a=a[:,255:767,255:767];v=v[255:767,255:767]
   with torch.no_grad():s=base.forward(model,a,'probe')[0].cpu().numpy()
   np.savez_compressed(D/'probes'/((row.get('product') or row['name'])+'_'+str(attempt)+'.npz'),scores=s,pred=s.argmax(0).astype(np.uint8),valid=v)
 probe(0);seeded();updates=0;skipped=0;actual=[]
 oldcounts={int(r['attempt']):r for r in read(OLD/'CROP_REPRODUCTION.csv')} if POOL=='CMC' else {int(r['attempt']):r for r in read(ROOT/POOL/'CROP_IDENTITY.csv')}
 for step in steps:
  row=rows[step['fit_index']];p=pseudo(POOL,row,step);cc=counts(p);ex=oldcounts[step['attempt']];assert cc==[int(ex[c]) for c in C];loss_value=gn=None;means=[None]*3;context('train')
  if sum(cc):
   a,_=image(POOL,row,step);optimizer.zero_grad(set_to_none=True);score=base.forward(model,a,'train');loss,ls=losses(score,p,ARM);assert torch.isfinite(loss);loss.backward();assert all(v.grad is None for n,v in model.named_parameters() if n not in names);assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
   gn=float(torch.sqrt(sum(q.grad.float().square().sum() for q in params)));optimizer.step();updates+=1;loss_value=float(loss.detach());means=[None if v is None else float(v.detach()) for v in ls]
  else:skipped+=1
  entry=dict(attempt=step['attempt'],fit_index=step['fit_index'],mask_sha=ah(p),Surface=cc[0],Cloud=cc[1],Shadow=cc[2],accepted=sum(cc),class_mean_CE=means,loss=loss_value,grad_norm=gn,updated=bool(sum(cc)),lr=1e-4);append(D/'TRAIN_CURVE.jsonl',entry)
  if step['attempt'] in ([100,500,1000] if POOL=='CMC' else [1000]):probe(step['attempt'])
  if step['attempt']%100==0:print(POOL,ARM,step['attempt'],updates,flush=True)
 assert updates==expected_updates and skipped==1000-updates;legacy.check_base(model,state,names);assert base.frozen_digest(model,names)==frozen
 torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names},D/'final_lora.pth')
 write(D/'ENDPOINT_LOCK.json',dict(status='sealed',pool=POOL,arm=ARM,attempts=1000,updates=updates,skipped=skipped,initial_sha=sha(D/'initial_lora.pth'),final_sha=sha(D/'final_lora.pth'),curve_sha=sha(D/'TRAIN_CURVE.jsonl'),base_unchanged=True,frozen_digest=frozen,sealed_epoch=time.time(),schedule_sha=sha(ROOT/POOL/'SCHEDULE.json')))
def evaluate():
 assert (ROOT/'ALL_TRAINED.json').exists();rows=inputs(POOL)['val'];context('evaluate');model,state,names,_=fresh(ARM!='Source')
 if ARM!='Source':
  assert sha(D/'final_lora.pth')==load(D/'ENDPOINT_LOCK.json')['final_sha'];delta=torch.load(D/'final_lora.pth',map_location='cpu')
  with torch.no_grad():
   for n,p in model.named_parameters():
    if n in delta:p.copy_(delta[n])
 result=[]
 for i,row in enumerate(rows):
  a,v=image(POOL,row)
  if POOL=='CMC':s=base.stitched(model,a,'evaluation_window')
  else:
   with torch.no_grad():s=base.forward(model,a,'evaluation_image')[0].cpu().numpy()
  p=s.argmax(0).astype(np.uint8);shadow=torch.from_numpy(s).softmax(0)[2].numpy();name=row.get('product') or row['name'];target=D/'evaluation'/(name+'.npz');np.savez_compressed(target,pred=p,valid=v,shadow=shadow)
  result.append(dict(name=name,file=str(target),sha256=sha(target),valid=int(v.sum()),shape=list(v.shape)))
  if (i+1)%100==0:print(POOL,ARM,'evaluate',i+1,flush=True)
 legacy.check_base(model,state,names);write(D/'EVALUATION_LOCK.json',dict(status='sealed',pool=POOL,arm=ARM,rows=result,sealed_epoch=time.time()))
if __name__=='__main__':
 try:startup();{'gates':gate_inputs,'teacher':teacher,'train':train,'evaluate':evaluate}[STAGE]()
 except BaseException:
  append(ROOT/'FAILURES.jsonl',dict(stage=STAGE,pool=POOL,arm=ARM,epoch=time.time(),traceback=traceback.format_exc()));raise
