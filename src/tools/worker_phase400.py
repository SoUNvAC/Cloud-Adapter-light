"""MMEngine paired training, fixed source/development evaluation, no hidden test loader."""
from phase400_common import *
import sys,traceback,copy
def configure(arm,kind):
 cfg=config(arm);cfg.load_from=str(SOURCE);cfg.work_dir=str(R/'work_dirs'/kind/arm);cfg.randomness=dict(seed=40001,deterministic=True);cfg.deterministic_warn_only=True;cfg.env_cfg.cudnn_benchmark=False
 cfg.train_dataloader.dataset.update(type='Phase400PairedDataset',prepared_manifest=str(R/'PREPARED_MANIFEST.json'),cohort='fit');cfg.train_dataloader.sampler=dict(type='Phase400PairedSampler',seed=40001);cfg.val_dataloader.dataset.prepared_manifest=str(R/'PREPARED_MANIFEST.json');cfg.test_dataloader=copy.deepcopy(cfg.val_dataloader);cfg.optim_wrapper.constructor='Phase400PEFTConstructor';cfg.custom_hooks=[dict(type='Phase400Hook')];cfg.val_evaluator=dict(type='Phase400Metric');cfg.test_evaluator=cfg.val_evaluator
 cfg.visualizer=dict(type='SegLocalVisualizer',vis_backends=[dict(type='LocalVisBackend')],name='phase400_visualizer');cfg.default_hooks.visualization=dict(type='SegVisualizationHook',draw=False)
 if kind=='short':cfg.train_cfg=dict(type='IterBasedTrainLoop',max_iters=120,val_interval=100000);cfg.default_hooks.checkpoint=dict(type='CheckpointHook',by_epoch=False,interval=100000,save_last=False,save_best=None)
 return cfg
def train(arm,kind):
 global budget
 torch=setup();os.environ['PHASE65_PREPARED_MANIFEST']=str(R/'PREPARED_MANIFEST.json');os.environ['PHASE65_SOURCE_CHECKPOINT']=str(SOURCE);import phase400_runtime as rt;rt.BUDGET=budget=Budget(kind+'_'+arm);rt.RUN_ARM=arm;rt.RUN_KIND=kind
 from mmengine.runner import Runner
 cfg=configure(arm,kind);assert not Path(cfg.work_dir).exists();save(R/'configs'/(kind+'_'+arm+'.json'),cfg.to_dict());runner=Runner.from_cfg(cfg);torch.use_deterministic_algorithms(True,warn_only=True)
 profile=None
 if kind=='short':
  from mmengine.hooks import Hook
  class Profile(Hook):
   priority='LOWEST'
   def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):profile.step()
  folder=R/'profiles';folder.mkdir(exist_ok=True);profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],schedule=torch.profiler.schedule(wait=20,warmup=1,active=3,repeat=1),on_trace_ready=lambda p:p.export_chrome_trace(str(folder/(arm+'.json'))),record_shapes=False,profile_memory=True);runner.register_hook(Profile());profile.start()
 runner.train()
 if profile is not None:
  profile.stop();save(R/'profiles'/(arm+'_KEYS.json'),[{'key':v.key,'count':v.count,'CPU_total_us':v.cpu_time_total,'CUDA_total_us':v.cuda_time_total,'self_CUDA_total_us':v.self_cuda_time_total} for v in profile.key_averages()])
 traces=[__import__('json').loads(l) for l in (Path(cfg.work_dir)/'STEP_TRACE.jsonl').read_text().splitlines()];assert len(traces)==(120 if kind=='short' else 4000);active=js(Path(cfg.work_dir)/'ACTIVE_SCOPE.json');sample=traces[20:] if kind=='short' else traces;elapsed=[v['seconds'] for v in sample];save(R/(kind+'_'+arm+'_COMPLETE.json'),{'passed':True,'steps':len(traces),'mean_step_seconds':sum(elapsed)/len(elapsed),'GPU_stream_ms_mean':sum(v['GPU_stream_ms'] for v in sample)/len(sample),'host_gap_mean':sum(v['host_gap_seconds'] for v in sample)/len(sample),'active_scope':active,'trace_SHA':sha(Path(cfg.work_dir)/'STEP_TRACE.jsonl'),'wall':time.monotonic()-budget.start,'work_dir':cfg.work_dir});budget.finish('passed')
def evaluate(arm,checkpoint,source_only=False,source_off=False,tag='Source'):
 global budget
 torch=setup();os.environ['PHASE65_PREPARED_MANIFEST']=str(R/'PREPARED_MANIFEST.json');os.environ['PHASE65_SOURCE_CHECKPOINT']=str(SOURCE);import phase400_runtime
 import numpy as np
 from mmseg.registry import DATASETS
 from mmengine.dataset import pseudo_collate
 budget=Budget('evaluate_'+tag);model,cfg,loading=build(arm,budget,baseline=source_only)
 if not source_only:
  ck=torch.load(checkpoint,map_location='cpu');assert ck['phase400']['Source_SHA']==EXPECTED_SOURCE;model.load_state_dict(ck['state_dict'],strict=True);model.backbone.set_target_enabled(not source_off)
 model.eval();output=R/'source_evaluations'/tag;assert not output.exists();output.mkdir(parents=True);source_cfg=copy.deepcopy(cfg.val_dataloader.dataset);source_cfg=dict(type='Phase64CloudSENParentDataset',data_root=js(R/'INPUT_LOCK.json')['source_val_root'],data_prefix=dict(img_path='img_dir/val',seg_map_path='ann_dir/val'),pipeline=[dict(type='LoadImageFromFile'),dict(type='Resize',scale=(512,512)),dict(type='LoadAnnotations'),dict(type='PackSegInputs')]);dataset=DATASETS.build(source_cfg);assert len(dataset)==535;records=[];aggregate=np.zeros((3,3),np.int64);h=model.register_forward_pre_hook(lambda m,a,k:budget.count('source_val_full_input'),with_kwargs=True);maximum=0.;different=0
 source_map={v['name']:v for v in js(R/'INPUT_LOCK.json')['source_val']}
 for i in range(len(dataset)):
  sample=dataset[i];name=Path(sample['data_samples'].img_path).name;ident=source_map[name];assert sha(ident['image'])==ident['image_SHA'] and sha(ident['mask'])==ident['mask_SHA'];budget.current={'tag':tag,'source_val':i,'name':name}
  with torch.inference_mode():d=model.test_step(pseudo_collate([sample]))[0]
  pred=d.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8);truth=sample['data_samples'].gt_sem_seg.data[0].numpy();valid=truth!=255;assert np.bincount(truth[valid],minlength=3).tolist()==ident['parent_counts'];cm=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3);aggregate+=cm;logits=d.seg_logits.data.cpu().float().numpy();rec={'name':name,'confusion':cm.tolist(),'prediction_SHA':hashlib.sha256(pred.tobytes()).hexdigest()}
  if source_only:
   target=output/(name+'.npy');np.save(target,logits);rec.update(logits_file=str(target.relative_to(R)),logits_SHA=sha(target))
  if source_off:
   base=np.load(R/'source_evaluations/Source'/(name+'.npy'),mmap_mode='r');delta=float(np.max(np.abs(logits-base)));maximum=max(maximum,delta);base._mmap.close();old=js(R/'source_evaluations/Source/METRICS.json')['per_input'][i];different+=int(rec['prediction_SHA']!=old['prediction_SHA']);rec['branch_off_max_abs']=delta;assert delta<=1e-5,('source off parity',name,delta)
  records.append(rec)
  if i%25==0:budget.check()
 h.remove();diag=aggregate.diagonal();union=aggregate.sum(0)+aggregate.sum(1)-diag;save(output/'METRICS.json',{'passed':True,'tag':tag,'source_branch':'off' if source_off else 'baseline' if source_only else 'on','checkpoint_SHA':sha(checkpoint),'state_SHA':state_sha(model.state_dict()),'mIoU':float(np.mean(diag/union)*100),'IoU':(diag/union*100).tolist(),'confusion':aggregate.tolist(),'per_input':records,'branch_off_max_abs':maximum if source_off else None,'branch_off_prediction_disagreement_images':different if source_off else None});budget.finish('passed')
 if source_only:
  import phase400_runtime as rt
  metric=rt.Phase400Metric();path=R/'evaluations/Source';path.mkdir(parents=True,exist_ok=False);metric.evaluation=str(path);ds=DATASETS.build(config('MH').val_dataloader.dataset);assert len(ds)==32;model.backbone.train(False);hh=model.register_forward_pre_hook(lambda m,a,k:None,with_kwargs=True)
  # This is a separate measured baseline target pass after the source stage.
  budget=Budget('evaluate_Source_development');root_hook=model.register_forward_pre_hook(lambda m,a,k:budget.count('development_full_input'),with_kwargs=True)
  for i in range(len(ds)):
   sample=ds[i]
   with torch.inference_mode():d=model.test_step(pseudo_collate([sample]))[0]
   metric.process(None,[d.to_dict()])
  metric.compute_metrics(metric.results);root_hook.remove();hh.remove();budget.finish('passed')
def main():
 if sys.argv[1] in ('short','formal'):train(sys.argv[2],sys.argv[1])
 else:evaluate(sys.argv[2],sys.argv[3],source_only=sys.argv[2]=='S',source_off=sys.argv[4]=='off',tag=sys.argv[5])
if __name__=='__main__':
 try:main()
 except BaseException:
  save('WORKER_FAILURE_'+sys.argv[1]+'_'+sys.argv[2]+'.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise
