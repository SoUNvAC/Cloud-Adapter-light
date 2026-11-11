"""Controlled continuation of450: frozen baseline versus task-module fine tuning."""
from pathlib import Path
import sys,os,json,hashlib,time,random,copy,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
ARM='A'
ROOT=Path('/home/scv/shared/phase454_20261106');R=ROOT/ARM/'network_run';P=Path('/home/scv/shared/phase431_20261102');SRC=Path('/home/scv/Cloud-Adapter-light/src')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code'];os.chdir(SRC)
from train_shadow450 import Images,digest,sha
from shadow_branch450 import LocalOcclusionBranch
from loss_shadow454 import confusion_ranking

def save(name,x):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def state_hash(items):
 h=hashlib.sha256()
 for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 from mmengine.config import Config
 from mmseg.registry import MODELS
 from mmseg.utils import register_all_modules
 from report_phase400 import describe
 R.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(45201);np.random.seed(45201);random.seed(45201);torch.backends.cudnn.benchmark=False
 register_all_modules();import phase431.integration
 cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
 teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True);teacher.cuda().eval().requires_grad_(False)
 student=copy.deepcopy(teacher) if ARM=='B' else teacher
 if ARM=='B':
  for n,p in student.named_parameters():p.requires_grad_(not n.startswith('backbone.encoder.'))
  for m in student.modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):m.requires_grad_(False)
 student.eval();teacher_hash=digest(teacher);encoder_hash=digest(student.backbone.encoder)
 bn_names=[n for n,m in student.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
 bn_items=lambda:[(n,t) for n,t in student.state_dict().items() if any(n.startswith(k+'.') for k in bn_names)]
 bn_hash=state_hash(bn_items());captured={}
 student.backbone.spatial.register_forward_hook(lambda m,a,o:captured.update(detail=o[1]));student.backbone.register_forward_hook(lambda m,a,o:captured.update(features=o))
 branch=LocalOcclusionBranch().cuda();origin=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth');branch.load_state_dict(torch.load(origin,map_location='cpu')['state_dict'],strict=True)
 active={n:p for n,p in student.named_parameters() if p.requires_grad};initial={n:p.detach().cpu().clone() for n,p in active.items()};initial_branch={n:p.detach().cpu().clone() for n,p in branch.named_parameters()}
 groups=[dict(params=list(branch.parameters()),lr=5e-5,peak=5e-5)]
 if active:groups.append(dict(params=list(active.values()),lr=2e-6,peak=2e-6))
 optimizer=torch.optim.AdamW(groups,weight_decay=.01);assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in [*branch.parameters(),*active.values()]}
 mean=torch.tensor([123.675,116.28,103.53],device='cuda')[None,:,None,None];std=torch.tensor([58.395,57.12,57.375],device='cuda')[None,:,None,None]
 train=Images('train');val=Images('val');assert len(train)==8490 and len(val)==535
 loader=DataLoader(train,batch_size=4,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(45201))
 vl=DataLoader(val,batch_size=4,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(45202))
 baseline=json.loads(Path('/home/scv/shared/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0];floor=np.array(baseline['IoU'][:2]);calls=0;start=time.monotonic();history=[];best_shadow=-1.;best_protected=-1.
 def scores(model,rgb):
  out=model.encode_decode((rgb-mean)/std,[dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512),padding_size=[0,0,0,0]) for _ in rgb]).clamp_min(1e-8)
  return (out/out.sum(1,keepdim=True)).log()
 def forward(rgb,training=False):
  nonlocal calls
  rgb=rgb.cuda(non_blocking=True).float()
  with torch.set_grad_enabled(training and ARM=='B'):base=scores(student,rgb)
  calls+=len(rgb);logits,delta,gate=branch(captured['detail'],captured['features'],rgb/255,base)
  if training and ARM=='B':
   with torch.no_grad():reference=scores(teacher,rgb)
   calls+=len(rgb)
  else:reference=base.detach()
  return logits,delta,gate,reference
 def checkpoint(step):
  state=dict(state_dict=branch.state_dict(),student_patch={n:t for n,t in student.state_dict().items() if not n.startswith('backbone.encoder.')},step=step,arm=ARM,origin_branch_sha=sha(origin))
  path=R/f'branch_{step}.pth';torch.save(state,path);save(f'weight_sha_{step}.json',dict(file=path.name,sha=sha(path),bytes=path.stat().st_size))
 def evaluate(step):
  nonlocal best_shadow,best_protected
  branch.eval();cm=np.zeros((3,3),np.int64);per=[]
  with torch.no_grad():
   for rgb,y,names in vl:
    logits,_,_,_=forward(rgb);pred=logits.argmax(1).cpu().numpy();truth=y.numpy()
    for t,p,n in zip(truth,pred,names):
     valid=t!=255;c=np.bincount(t[valid].astype(np.int64)*3+p[valid],minlength=9).reshape(3,3);cm+=c;per.append(dict(name=n,confusion=c.tolist()))
  metrics=describe(cm);protected=bool(np.all(np.array(metrics['IoU'][:2])>=floor));success=protected and metrics['IoU'][2]>=70
  if step==0:
   assert max(abs(a-b) for a,b in zip(metrics['IoU'],baseline['IoU']))<.002
   protected=True # exact origin checkpoint, numerical batch4 differences only
  row=dict(step=step,**metrics,protected=protected,success=success,seconds=time.monotonic()-start);checkpoint(step)
  if step>0 and metrics['IoU'][2]>best_shadow:best_shadow=metrics['IoU'][2];save('BEST_SHADOW.json',row)
  if protected and metrics['IoU'][2]>best_protected:best_protected=metrics['IoU'][2];save('BEST_PROTECTED.json',row)
  history.append(row);save('HISTORY.json',history);save(f'validation/step_{step}.json',dict(metrics=row,per_image=per));branch.train();return row
 save('LOCK.json',dict(method='Original450 loss plus 0.1 image/class-balanced hardest1024 shadow-vs-surface/cloud margin ranking; margin1; real source only; no synthetic images',arm=ARM,origin=str(origin),origin_sha=sha(origin),steps=6000,batch_size=4,seed=45201,branch_lr=5e-5,task_lr=2e-6 if ARM=='B' else 0,branch_parameters=sum(p.numel() for p in branch.parameters()),task_parameters=sum(p.numel() for p in active.values()),active_task_names=list(active),target_label_supervision_percent=0,reference='Frozen original431 for all hard-error masks/KL/softIoU; unchanged across arms',BN='All BN statistics and affine parameters fixed; all base modules eval; DINOv3 frozen with gradient flow through its blocks in B',selection='Best trained shadow plus best nonshadow-protected relative to450; origin step0 is fallback',code_sha={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('loss_shadow454.py'),Path('/home/scv/shared/phase450_20261106/shadow_branch450.py')]}))
 save('INITIAL_REPRODUCTION.json',evaluate(0));iterator=iter(loader)
 for step in range(1,6001):
  try:rgb,y,names=next(iterator)
  except StopIteration:iterator=iter(loader);rgb,y,names=next(iterator)
  if step<=10 or step%1000==0:
   with (R/'INPUT_TRACE.jsonl').open('a') as h:h.write(json.dumps(dict(step=step,names=list(names),batch_sha=state_hash([('rgb',rgb),('label',y)])))+'\n')
  y=y.cuda(non_blocking=True);logits,delta,gate,reference=forward(rgb,True);valid=y!=255;old=reference.argmax(1);hard=(old!=y)&((old==2)|(y==2))&valid
  weights=torch.ones_like(y,dtype=torch.float);weights[hard]=4.;weights[(y==2)&(old==2)]=1.5
  ce=F.cross_entropy(logits,y,ignore_index=255,reduction='none');loss_ce=(ce*weights*valid).sum()/(weights*valid).sum()
  prob=logits.softmax(1);oldprob=reference.softmax(1);keep=valid&(old==y)&(y!=2);kl=(oldprob*(reference-logits.log_softmax(1))).sum(1);loss_keep=(kl*keep).sum()/keep.sum().clamp_min(1);softloss=0.
  for c in [0,1]:
   gt=(y==c)&valid;p=prob[:,c]*valid;q=oldprob[:,c]*valid
   softloss=softloss+F.relu((q*gt).sum()/((q+gt-q*gt).sum()+1)-(p*gt).sum()/((p+gt-p*gt).sum()+1))
  loss=loss_ce+2*loss_keep+2*softloss+.01*delta.square().mean();assert torch.isfinite(loss)
  rank,rank_info=confusion_ranking(logits,y)
  loss=loss+.1*rank
  if step<=10 or step%1000==0:save(f'ranking/step_{step}.json',rank_info)
  for g in optimizer.param_groups:g['lr']=g['peak']*min(step/200,1)*(1-step/6001)**.9
  optimizer.zero_grad(set_to_none=True);loss.backward()
  for g in optimizer.param_groups:torch.nn.utils.clip_grad_norm_(g['params'],1.)
  optimizer.step()
  if step in [10,100] or step%1000==0:
   assert digest(teacher)==teacher_hash and digest(student.backbone.encoder)==encoder_hash and state_hash(bn_items())==bn_hash
   assert all(p.grad is None for p in student.backbone.encoder.parameters())
   changed={g:sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in active.items() if n.startswith(g)) for g in ['backbone.spatial.','backbone.interactions.','backbone.refinement.','backbone.local.','backbone.output_residual.','decode_head.']}
   if ARM=='B':assert all(changed.values()),changed
   assert any(not torch.equal(initial_branch[n],p.detach().cpu()) for n,p in branch.named_parameters())
   save('UPDATE_CHECK.json',dict(step=step,reference_unchanged=True,encoder_unchanged=True,BN_unchanged=True,changed=changed,branch_updated=True,peak_memory_bytes=torch.cuda.max_memory_allocated()))
  if step%50==0:save('STATUS.json',dict(step=step,total=6000,loss=float(loss.detach()),seconds=time.monotonic()-start,forward_images=calls))
  if step%1000==0:evaluate(step)
 save('COMPLETE.json',dict(passed=True,steps=6000,seconds=time.monotonic()-start,forward_images=calls,best_shadow=json.loads((R/'BEST_SHADOW.json').read_text()),best_protected=json.loads((R/'BEST_PROTECTED.json').read_text())))
if __name__=='__main__':
 try:main()
 except BaseException:
  R.mkdir(parents=True,exist_ok=True);save('FAILURE.json',dict(error=traceback.format_exc()));raise
