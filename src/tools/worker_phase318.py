"""Unique fixed selective-CB arms. Reuse frozen317 model/reader/evaluation functions."""
import gc,random,traceback
import numpy as np
import torch
from torch.nn import functional as F
from phase318_common import *
from phase318_selectors import selection,masses,product,crop
from phase317_images import image,pseudo
import worker_phase317 as old
base=old.base;legacy=old.legacy
STAGE,POOL,ARM=sys.argv[1:4];D=ROOT/POOL/ARM
def call(kind):
 p=ROOT/'CALLS.jsonl';count=sum(1 for _ in p.open()) if p.exists() else 0;assert count<11000
 append(p,dict(stage=STAGE,pool=POOL,arm=ARM,context=old.CONTEXT,kind=kind,epoch=time.time(),PID=os.getpid()))
old.ROOT=ROOT;old.D=D;old.POOL=POOL;old.ARM=ARM;old.STAGE=STAGE;old.call=call
def weighted(s,p,selected):
 meta=masses(p,selected);assert meta['N']>0
 target=torch.from_numpy(p.astype(np.int64))[None].cuda();valid=target!=255;pick=torch.from_numpy(selected)[None].cuda()
 pix=F.cross_entropy(s,target,ignore_index=255,reduction='none');w=torch.zeros_like(pix)
 for c in range(3):w[target==c]=meta['b'][c]
 w[(target==2)&~pick]=1.;w=w.detach();den=w.sum();loss=(pix*w).sum()/den
 ce=[None if not (target==c).any() else float(pix[target==c].mean().detach()) for c in range(3)]
 wm=[float(w[target==c].sum()) for c in range(3)];num=[float((w*pix)[target==c].sum().detach()) for c in range(3)]
 assert np.allclose(wm,meta['class_mass'],rtol=1e-6,atol=1e-6) and abs(float(den)-meta['denominator'])<=1e-6*meta['denominator']
 return loss,dict(**meta,class_mean_CE=ce,actual_class_mass=wm,weighted_CE_sum=num,actual_denominator=float(den))
def toy():
 checks=[]
 cases=[([0,0,1,2,2],[0,0,0,1,0]),([2,2,2],[1,0,0]),([0,1,1],[0,0,0]),([0,1,2],[0,0,0]),([0,1,2],[0,0,1]),([255,255],[0,0])]
 for values,selected in cases:
  p=np.array([values],np.uint8);g=np.array([selected],bool);m=masses(p,g)
  if not m['N']:checks.append(dict(empty=True,skip=True));continue
  x=torch.arange(3*p.size,dtype=torch.float32,device='cuda').reshape(1,3,1,p.size)/7;x.requires_grad_();loss,logged=weighted(x,p,g)
  y=torch.tensor(values,device='cuda')[None,None];ce=F.cross_entropy(x,y,ignore_index=255,reduction='none').detach().cpu().numpy()[0]
  manual=np.zeros(p.shape,np.float64)
  for c in range(3):manual[p==c]=m['b'][c]
  manual[(p==2)&~g]=1.;expected=float((manual*ce).sum()/manual.sum());assert abs(float(loss)-expected)<=1e-6
  full=p==2;full_loss,_=weighted(x,p,full);cb,_=old.losses(x,p,'CB');err=float(abs(full_loss-cb));graderr=float((torch.autograd.grad(full_loss,x,retain_graph=True)[0]-torch.autograd.grad(cb,x)[0]).abs().max());assert err<=1e-6 and graderr<=1e-6
  checks.append(dict(pseudo=values,selected=selected,manual_loss=expected,loss_error=abs(float(loss)-expected),full_CB_loss_error=err,full_CB_gradient_error=graderr,weights=m))
 # Explicit tie: original indices dominate training orientation.
 q=np.array([.95,.95,.96,.95],np.float32);ids=np.arange(4);assert np.lexsort((ids,-q))[:2].tolist()==[2,0]
 assert ids[::-1][np.lexsort((ids[::-1],-q[::-1]))[:2]].tolist()==[2,0]
 a=np.arange(3*512*512).reshape(3,512,512);assert np.array_equal(a,a[:,:,::-1][:,:,::-1])
 return dict(status='passed',manual=checks,original_index_ties=True,double_flip_exact=True,no_shadow_not_necessarily_PL=True,majority_shadow_CB_can_be_below_one=True)
def teacher():
 assert ARM=='Source' and not (ROOT/POOL/'FLIP_TEACHER_LOCK.json').exists();old.context('flip_teacher');model,state,names,_=old.fresh(False)
 out=[];target=ROOT/POOL/'flip_teacher';target.mkdir()
 for i,row in enumerate(inputs(POOL)['fit']):
  a,v=image(POOL,row);assert np.array_equal(a,a[:,:,::-1][:,:,::-1])
  if POOL=='CMC':
   from phase302_images import WINDOWS
   sums=np.zeros((3,1022,1022),np.float32);cover=np.zeros((1022,1022),np.uint8);original_cover=np.zeros_like(cover)
   with torch.no_grad():
    for t,l in WINDOWS:
     window=a[:,t:t+512,l:l+512];s=base.forward(model,window[:,:,::-1].copy(),'flip_teacher_window')[0].cpu().numpy()[:,:,::-1]
     sums[:,t:t+512,l:l+512]+=s;cover[t:t+512,l:l+512]+=1;original_cover[t:t+512,l:l+512]+=1
   assert (cover>0).all() and np.array_equal(cover,original_cover);s=sums/cover[None]
  else:
   with torch.no_grad():s=base.forward(model,a[:,:,::-1].copy(),'flip_teacher_image')[0].cpu().numpy()[:,:,::-1].copy()
  total=s.sum(0);q=np.divide(s[2],total,out=np.zeros_like(total),where=total>1e-12);pred=s.argmax(0).astype(np.uint8)
  # Image valid after input flip and output inverse is exactly original valid.
  vv=v[:,::-1][:,::-1].copy();assert np.array_equal(vv,v)
  orig=OLD/'teacher'/(name(row)+'.npz') if POOL=='CMC' else PRIOR/POOL/'teacher'/(name(row)+'.npz')
  with np.load(orig) as z:p=z['pseudo'];before=z['pred'];assert np.array_equal(z['valid'],v)
  g=(p==2)&vv&(pred==2)&(q>=np.float32(.9));path=target/(name(row)+'.npz');np.savez_compressed(path,q_shadow=q,pred=pred,valid=vv)
  if i==0:np.savez_compressed(ROOT/POOL/'FLIP_SCORE_GATE.npz',scores=s,valid=v,window_cover=cover if POOL=='CMC' else np.ones(v.shape,np.uint8))
  transitions=np.bincount((3*before[v]+pred[v]).astype(np.int64),minlength=9).reshape(3,3)
  out.append(dict(name=name(row),file=str(path),sha256=sha(path),original_shadow=int((p==2).sum()),selected=int(g.sum()),selected_sha256=ah(g),valid=int(v.sum()),transition=transitions.tolist()))
  if (i+1)%10==0:print(POOL,'flip teacher',i+1,flush=True)
 legacy.check_base(model,state,names);write(ROOT/POOL/'FLIP_TEACHER_LOCK.json',dict(status='sealed',rows=out,original_view_reused=True,truth_read=False))
def seal_selection():
 out=[];poolinfo={}
 for pool in ['CMC','SPARCS','L8']:
  assert (ROOT/pool/'FLIP_TEACHER_LOCK.json').exists();same=True;full=True;updates=0;counts=np.zeros(3,np.int64)
  prior={int(r['attempt']):r for r in read(OLD/'CROP_REPRODUCTION.csv' if pool=='CMC' else PRIOR/pool/'CROP_IDENTITY.csv')}
  for st in schedule(pool):
   p,g,c=selection(pool,st);m=masses(p,g);e=prior[st['attempt']];assert m['counts']==[int(e[k]) for k in C];assert np.array_equal(p,pseudo(pool,inputs(pool)['fit'][st['fit_index']],st))
   cm=masses(p,c);assert m==cm;same&=bool(np.array_equal(g,c));full&=bool(np.array_equal(g,p==2));updates+=bool(m['N']);counts+=np.array(m['counts'])>0
   out.append(dict(pool=pool,**st,**m,pseudo_sha=ah(p),Flip_sha=ah(g),Conf_sha=ah(c),identical=bool(np.array_equal(g,c)),intersection=int((g&c).sum()),union=int((g|c).sum()),A_Sh=m['counts'][2],selected_fraction=m['k']/m['counts'][2] if m['counts'][2] else None))
  assert updates=={'CMC':955,'SPARCS':1000,'L8':951}[pool]
  poolinfo[pool]=dict(identical_all_steps=same,full_all_steps=full,updates=updates,skipped=1000-updates,class_present_steps=counts.tolist(),mode='old_CB_equivalence_gate_required' if full else 'single_identical_arm' if same else 'two_distinct_arms')
  product.cache_clear()
 table(ROOT/'SELECTION_AND_WEIGHT.csv',out);write(ROOT/'SELECTION_LOCK.json',dict(status='sealed',pools=poolinfo,table_sha=sha(ROOT/'SELECTION_AND_WEIGHT.csv'),teacher={p:sha(ROOT/p/'FLIP_TEACHER_LOCK.json') for p in poolinfo},no_truth_read=True))
def train():
 info=load(ROOT/'SELECTION_LOCK.json')['pools'][POOL]
 assert (POOL,ARM) in ARMS and not (D/'initial_lora.pth').exists();rows=inputs(POOL)['fit'];steps=schedule(POOL)
 if ARM=='Conf-CB' and info['identical_all_steps']:
  src=ROOT/POOL/'Flip-CB';e=src/('EQUIVALENT_ARM.json' if (src/'EQUIVALENT_ARM.json').exists() else 'ENDPOINT_LOCK.json');assert e.exists();target=load(e)['target'] if e.name=='EQUIVALENT_ARM.json' else str(src);write(D/'EQUIVALENT_ARM.json',dict(status='strict_equivalent',pool=POOL,arm=ARM,target=target,reason='1000 selection masks and objectives exactly identical before training',selection_lock_sha=sha(ROOT/'SELECTION_LOCK.json'),endpoint_lock_sha=sha(e)));return
 first=next(st for st in steps if masses(selection(POOL,st)[0],selection(POOL,st)[1])['N']);a,_=image(POOL,rows[first['fit_index']],first);old.context('train_gate');source,state,_,_=old.fresh(False)
 with torch.no_grad():s=base.forward(source,a,'gate_Source_original',original=True)
 del source;gc.collect();torch.cuda.empty_cache();model,state,names,cfg=old.fresh(True);params=[p for n,p in model.named_parameters() if n in names];assert sum(p.numel() for p in params)==294912
 initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names};prior=torch.load(OLD/'initial_lora.pth',map_location='cpu');assert set(prior)==set(initial) and all(torch.equal(prior[n],v) for n,v in initial.items())
 with torch.no_grad():z=base.forward(model,a,'gate_zero_lora');err=float((z-s).abs().max());assert err<=1e-6
 frozen=base.frozen_digest(model,names);assert frozen==load(OLD/'ENDPOINT_LOCK.json')['base_initial_final_sha256'];torch.save(initial,D/'initial_lora.pth')
 optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8);p,g,c=selection(POOL,first);score=base.forward(model,a,'gate_gradient');loss,_=weighted(score,p,g if ARM=='Flip-CB' else c)
 equivalent=None
 if info['full_all_steps']:
  cb,_=old.losses(score,p,'CB');ga=torch.autograd.grad(loss,params,retain_graph=True);gb=torch.autograd.grad(cb,params,retain_graph=True);le=float(abs(loss-cb));ge=max(float((x-y).abs().max()) for x,y in zip(ga,gb));assert le<=1e-6 and ge<=1e-6
  aa=[torch.nn.Parameter(v.detach().clone()) for v in params];bb=[torch.nn.Parameter(v.detach().clone()) for v in params]
  for xs,grads in [(aa,ga),(bb,gb)]:
   opt=torch.optim.AdamW(xs,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
   for x,grad in zip(xs,grads):x.grad=grad.detach().clone()
   opt.step()
  ue=max(float((x-y).abs().max()) for x,y in zip(aa,bb));assert ue<=1e-6;equivalent=dict(loss_error=le,gradient_error=ge,update_error=ue,original_CB_function_sha=sha(R/'src/tools/worker_phase317.py'))
 loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params) and all(p.grad is None for n,p in model.named_parameters() if n not in names)
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in initial:p.copy_(initial[n])
 model.zero_grad(set_to_none=True);old.seeded();assert not optimizer.state and all(p.grad is None for p in params);legacy.check_base(model,state,names)
 cfg.dump(str(D/'resolved_config.py'));write(D/'MODEL_ACCEPTANCE.json',dict(status='passed',zero_lora_error=err,fresh_initial_equals304=True,initial_tensor_SHA={n:ah(v.numpy()) for n,v in initial.items()},B_zero=True,parameters=294912,only_lora_grad=True,initial_optimizer_grad_RNG_restored=True,frozen_digest=frozen,seed=30201))
 if equivalent is not None:
  target=PRIOR/POOL/'CB';assert load(target/'MODEL_ACCEPTANCE.json')['initial_tensor_SHA']=={n:ah(v.numpy()) for n,v in initial.items()};write(D/'OLD_CB_EQUIVALENCE.json',equivalent);write(D/'EQUIVALENT_ARM.json',dict(status='strict_equivalent',pool=POOL,arm=ARM,target=str(target),reason='all steps select every Shadow; oldCB loss/gradient/AdamW update gated <=1e-6 from same initial',selection_lock_sha=sha(ROOT/'SELECTION_LOCK.json'),endpoint_lock_sha=sha(target/'ENDPOINT_LOCK.json')));return
 def probe(attempt):
  old.context('probe_'+str(attempt));probes=sorted(rows,key=name)[:4 if POOL=='CMC' else 1]
  for row in probes:
   a,v=image(POOL,row)
   if POOL=='CMC':a=a[:,255:767,255:767];v=v[255:767,255:767]
   with torch.no_grad():s=base.forward(model,a,'probe')[0].cpu().numpy()
   np.savez_compressed(D/'probes'/(name(row)+'_'+str(attempt)+'.npz'),scores=s,pred=s.argmax(0).astype(np.uint8),valid=v)
 probe(0);old.seeded();updates=0
 locked={int(r['attempt']):r for r in read(ROOT/'SELECTION_AND_WEIGHT.csv') if r['pool']==POOL}
 for st in steps:
  p,g,c=selection(POOL,st);selected=g if ARM=='Flip-CB' else c;m=masses(p,selected);e=locked[st['attempt']];assert ah(p)==e['pseudo_sha'] and ah(selected)==e['Flip_sha' if ARM=='Flip-CB' else 'Conf_sha'];old.context('train');extra={};lv=gn=None
  if m['N']:
   a,_=image(POOL,rows[st['fit_index']],st);optimizer.zero_grad(set_to_none=True);loss,extra=weighted(base.forward(model,a,'train'),p,selected);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for n,p in model.named_parameters() if n not in names) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
   gn=float(torch.sqrt(sum(p.grad.float().square().sum() for p in params)));optimizer.step();updates+=1;lv=float(loss.detach())
  append(D/'TRAIN_CURVE.jsonl',dict(attempt=st['attempt'],fit_index=st['fit_index'],mask_sha=ah(p),selected_sha=ah(selected),**m,logged=extra,loss=lv,grad_norm=gn,updated=bool(m['N']),lr=1e-4))
  if st['attempt'] in ([100,500,1000] if POOL=='CMC' else [1000]):probe(st['attempt'])
  if st['attempt']%100==0:print(POOL,ARM,st['attempt'],updates,flush=True)
 assert updates==info['updates'];legacy.check_base(model,state,names);assert base.frozen_digest(model,names)==frozen
 torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names},D/'final_lora.pth');write(D/'ENDPOINT_LOCK.json',dict(status='sealed',pool=POOL,arm=ARM,attempts=1000,updates=updates,skipped=1000-updates,initial_sha=sha(D/'initial_lora.pth'),final_sha=sha(D/'final_lora.pth'),curve_sha=sha(D/'TRAIN_CURVE.jsonl'),base_unchanged=True,frozen_digest=frozen,sealed_epoch=time.time(),selection_lock_sha=sha(ROOT/'SELECTION_LOCK.json')))
def evaluate():
 if (D/'EQUIVALENT_ARM.json').exists():
  ref=load(D/'EQUIVALENT_ARM.json');e=Path(ref['target'])/'EVALUATION_LOCK.json';assert e.exists();write(D/'EVALUATION_REFERENCE.json',dict(target=str(e),sha256=sha(e)));return
 old.evaluate()
if __name__=='__main__':
 try:
  old.startup();resource_check()
  def no_reference(event,args):
   if event=='open' and isinstance(args[0],(str,bytes)):
    p=os.fsdecode(args[0]).replace('\\','/').lower();assert not any(t in p for t in ['/reference/','_fixedmask.tif','selector_quality','damage_benefit','error_flow','per_scene_metrics','metrics.csv']),p
  sys.addaudithook(no_reference)
  if STAGE=='gates':write(ROOT/'MATHEMATICAL_ACCEPTANCE.json',toy())
  else:{'teacher':teacher,'selection':seal_selection,'train':train,'evaluate':evaluate}[STAGE]()
 except BaseException:append(ROOT/'FAILURES.jsonl',dict(stage=STAGE,pool=POOL,arm=ARM,epoch=time.time(),traceback=traceback.format_exc()));raise
