"""MMEngine training components with paired augmentation, scope and checkpoint guards."""
from phase400_common import *
import copy,random,math
import numpy as np,torch
from mmengine.hooks import Hook
from mmengine.evaluator import BaseMetric
from mmseg.registry import DATASETS,DATA_SAMPLERS,METRICS,HOOKS
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from cloud_adapter.datasets.phase65_catalogue import Phase65CatalogueDataset
from cloud_adapter.optimizers.peft_optimizer_constructor import PEFTOptimWrapperConstructor
from torch.utils.data import Sampler
BUDGET=None;RUN_ARM=None;RUN_KIND=None

@DATASETS.register_module()
class Phase400PairedDataset(Phase65CatalogueDataset):
 def __getitem__(self,key):
  index,draw=key;state=(random.getstate(),np.random.get_state(),torch.get_rng_state());seed=int.from_bytes(hashlib.sha256(f'4000103:{draw}'.encode()).digest()[:4],'little');random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  try:v=super().__getitem__(index);v['data_samples'].set_metainfo({'phase400_draw':draw,'phase400_augmentation_seed':seed});return v
  finally:random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])

@DATA_SAMPLERS.register_module()
class Phase400PairedSampler(Sampler):
 def __init__(self,dataset,shuffle=True,seed=40001,round_up=True,cursor=0):self.n=len(dataset);self.seed=seed;self.cursor=cursor
 def __iter__(self):
  g=torch.Generator().manual_seed(self.seed);draw=0
  while True:
   for index in torch.randperm(self.n,generator=g).tolist():
    if draw>=self.cursor:yield index,draw
    draw+=1
 def __len__(self):return self.n
 def set_epoch(self,epoch):pass

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class Phase400PEFTConstructor(PEFTOptimWrapperConstructor):
 def __call__(self,model):
  model.train();return super().__call__(model)

@METRICS.register_module()
class Phase400Metric(BaseMetric):
 def __init__(self,**kwargs):super().__init__(**kwargs);self.evaluation=None;self.counter=0
 def process(self,data_batch,data_samples):
  for d in data_samples:
   truth=d['gt_sem_seg']['data'][0].detach().cpu().numpy().astype(np.uint8);pred=d['pred_sem_seg']['data'][0].detach().cpu().numpy().astype(np.uint8);assert truth.shape==pred.shape;valid=truth!=255;mat=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3);logits=d['seg_logits']['data'].detach().float().clamp_min(0);probs=logits/logits.sum(0,keepdim=True).clamp_min(1e-12);assert torch.isfinite(probs).all().item();product=Path(d['img_path']).name.removesuffix('_rgb.npy');row=next(r for r in js(R/'INPUT_LOCK.json')['target_rows'] if r['product']==product and r['split']=='development_val');assert np.bincount(truth[valid],minlength=3).tolist()==row['label_counts'];path=Path(self.evaluation)/(product+'.npz');assert not path.exists();np.savez(path,shadow=probs[2].cpu().numpy(),truth=truth);self.results.append({'product':product,'group_id':row['group_id'],'confusion':mat.tolist(),'prediction_SHA':hashlib.sha256(pred.tobytes()).hexdigest(),'score_file':str(path.relative_to(R)),'score_SHA':sha(path)});self.counter+=1
 def compute_metrics(self,results):
  assert len(results)==32;mat=np.sum([v['confusion'] for v in results],axis=0);d=mat.diagonal();u=mat.sum(0)+mat.sum(1)-d;iou=d/u;value=float(np.mean(iou)*100);save(Path(self.evaluation)/'METRICS.json',{'mIoU':value,'IoU':(iou*100).tolist(),'confusion':mat.tolist(),'per_group':results,'probability':'nonnegative semantic score normalized by channel sum, no extra softmax; argmax unchanged','primary_metric':'pooled 3 class mIoU'});return {'mIoU':value}

@HOOKS.register_module()
class Phase400Hook(Hook):
 priority='VERY_HIGH'
 def before_train(self,runner):
  self.started=time.monotonic();self.last=time.monotonic();self.save_times=[];model=runner.model;model.train();permitted=('backbone.target_msre.','backbone.target_head_delta.');self.frozen={n:tensor_sha(t) for n,t in model.state_dict().items() if not n.startswith(permitted)};self.source_frozen=dict(self.frozen);self.file=(Path(runner.work_dir)/'STEP_TRACE.jsonl').open('x',buffering=1);self.reference=None
  original_save=runner.save_checkpoint
  def timed_save(*args,**kwargs):
   begin=time.monotonic();value=original_save(*args,**kwargs);self.save_times.append(time.monotonic()-begin);return value
  runner.save_checkpoint=timed_save
  if RUN_ARM!='MH':self.reference=(R/'work_dirs'/RUN_KIND/'MH/STEP_TRACE.jsonl').open()
  active={n:p.numel() for n,p in model.named_parameters() if p.requires_grad};expected={'MH':380577,'M':354433,'H':26144}[RUN_ARM];assert sum(active.values())==expected and all(n.startswith(permitted) for n in active);params=[p for g in runner.optim_wrapper.optimizer.param_groups for p in g['params']];assert {id(p) for p in params}=={id(p) for p in model.parameters() if p.requires_grad};save(Path(runner.work_dir)/'ACTIVE_SCOPE.json',{'names':active,'count':expected,'frozen_SHA':self.frozen,'MsRE_initial_SHA':state_sha(model.backbone.target_msre.state_dict()),'head_initial_SHA':state_sha(model.backbone.target_head_delta.state_dict())});self.grad_seen={}
  for n,p in model.named_parameters():
   if p.requires_grad:
    def grad(g,name=n):
     assert torch.isfinite(g).all().item(),('nonfinite gradient',name);self.grad_seen[name]=True;return g
    p.register_hook(grad)
  self.call_hook=model.register_forward_pre_hook(lambda m,a,k:BUDGET.count('train_full_input'),with_kwargs=True)
 def before_train_iter(self,runner,batch_idx,data_batch=None):
  self.begin=time.monotonic();img=data_batch['inputs'][0];ds=data_batch['data_samples'][0];truth=ds.gt_sem_seg.data;self.current={'step':runner.iter+1,'draw':ds.metainfo['phase400_draw'],'product':Path(ds.img_path).name,'image_SHA':tensor_sha(img),'label_SHA':tensor_sha(truth),'augmentation_seed':ds.metainfo['phase400_augmentation_seed'],'host_gap_seconds':self.begin-self.last};assert self.current['draw']==runner.iter;BUDGET.current={'arm':RUN_ARM,'kind':RUN_KIND,**self.current};self.cuda_start=torch.cuda.Event(enable_timing=True);self.cuda_end=torch.cuda.Event(enable_timing=True);self.cuda_start.record()
  if self.reference:
   ref=json.loads(self.reference.readline());assert all(ref[k]==self.current[k] for k in ('step','draw','product','image_SHA','label_SHA','augmentation_seed')),('paired input mismatch',ref,self.current)
 def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
  BUDGET.updates+=1;losses={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in outputs.items()};assert all(math.isfinite(v) for v in losses.values());self.cuda_end.record();self.cuda_end.synchronize();self.current.update(seconds=time.monotonic()-self.begin,GPU_stream_ms=self.cuda_start.elapsed_time(self.cuda_end),losses=losses,LR=[g['lr'] for g in runner.optim_wrapper.optimizer.param_groups]);self.file.write(json.dumps(self.current)+'\n');self.last=time.monotonic()
  if (runner.iter+1)%100==0 or runner.iter==0:
   now={n:tensor_sha(t) for n,t in runner.model.state_dict().items() if n in self.frozen};assert now==self.frozen,'frozen Source parameter/buffer changed';assert all(p.grad is None for n,p in runner.model.named_parameters() if not p.requires_grad);save(Path(runner.work_dir)/f'FREEZE_{runner.iter+1}.json',{'passed':True,'Source_state_SHA':hashlib.sha256(json.dumps(now,sort_keys=True).encode()).hexdigest(),'active_gradient_tensors_seen':len(self.grad_seen),'step':runner.iter+1});BUDGET.check()
 def before_val(self,runner):
  self.call_hook.remove();self.eval_path=R/'evaluations'/RUN_ARM/('step_'+str(runner.iter));assert not self.eval_path.exists();self.eval_path.mkdir(parents=True);runner.val_evaluator.metrics[0].evaluation=str(self.eval_path);self.eval_hook=runner.model.register_forward_pre_hook(lambda m,a,k:BUDGET.count('development_full_input'),with_kwargs=True)
 def after_val(self,runner):
  self.eval_hook.remove();self.call_hook=runner.model.register_forward_pre_hook(lambda m,a,k:BUDGET.count('train_full_input'),with_kwargs=True)
 def before_save_checkpoint(self,runner,checkpoint):
  cursor=self.current['step'];checkpoint['phase400']={'arm':RUN_ARM,'kind':RUN_KIND,'Source_SHA':js(R/'INPUT_LOCK.json')['Source_SHA'],'input_lock_SHA':sha(R/'INPUT_LOCK.json'),'draw_cursor':cursor,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'sampler':'deterministic randperm + draw-indexed augmentation; exact cursor replay; CUDA kernels may remain nondeterministic'};checkpoint.setdefault('optimizer',runner.optim_wrapper.state_dict());checkpoint.setdefault('param_schedulers',[s.state_dict() for s in runner.param_schedulers])
 def after_train(self,runner):
  self.call_hook.remove();self.file.close()
  if self.reference:self.reference.close()
  save(Path(runner.work_dir)/'SAVE_TIMING.json',{'seconds':self.save_times,'total':sum(self.save_times)})
