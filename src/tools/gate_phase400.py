"""Declared compatibility/freeze/gradient/restore gates before the three short runs."""
from phase400_common import *
import sys,traceback,copy,gc,random
def main():
 global budget
 torch=setup();import numpy as np;os.environ['PHASE65_PREPARED_MANIFEST']=str(R/'PREPARED_MANIFEST.json');os.environ['PHASE65_SOURCE_CHECKPOINT']=str(SOURCE);budget=Budget('correctness_gate');assert sha(SOURCE)==EXPECTED_SOURCE
 import phase400_runtime as rt
 from mmseg.registry import DATASETS
 from mmengine.dataset import pseudo_collate
 from audit_phase52_sparse_msre import logits
 from audit_phase64_model_matrix import parameter_report,scope_gate
 from mmengine.optim import build_optim_wrapper
 from mmengine.optim.scheduler import LinearLR,PolyLR
 # Reuse existing sparse-feature tests, without any historic launcher or old test pool.
 import importlib.util
 spec=importlib.util.spec_from_file_location('phase400_legacy_unit',SRC/'tests/test_phase52_sparse_msre.py');unit=importlib.util.module_from_spec(spec);spec.loader.exec_module(unit)
 for n in ('test_sparse_msre_disabled_is_exact_identity','test_sparse_msre_only_changes_registered_blocks','test_phase52_parameter_budget_for_dinov2_small'):getattr(unit,n)()
 dataset=DATASETS.build(dict(type='Phase400PairedDataset',prepared_manifest=str(R/'PREPARED_MANIFEST.json'),cohort='fit',pipeline=config('MH').train_pipeline));sample=dataset[0,0];batch=pseudo_collate([sample]);source,cfg,_=build('MH',budget,baseline=True);source.eval();pre=source.data_preprocessor(batch,False);x=pre['inputs'];assert x.shape==(1,3,512,512)
 budget.count('gate_Source_baseline')
 with torch.inference_mode():reference=logits(source,x).detach().cpu()
 del source;gc.collect();torch.cuda.empty_cache();reports=[];initials={};gate_samples=[]
 for arm in ('MH','M','H'):
  start=time.monotonic();model,cfg,load=build(arm,budget);total,active,names=parameter_report(model);assert active=={'MH':380577,'M':354433,'H':26144}[arm] and scope_gate('msre',names,active);frozen={n:tensor_sha(t) for n,t in model.state_dict().items() if not n.startswith(('backbone.target_msre.','backbone.target_head_delta.'))};initial={'M':state_sha(model.backbone.target_msre.state_dict()),'H':state_sha(model.backbone.target_head_delta.state_dict())};initials[arm]=initial
  model.eval();model.backbone.set_target_enabled(False);budget.count('gate_'+arm+'_off')
  with torch.inference_mode():off=logits(model,x).detach().cpu()
  parity=float((off-reference).abs().max());assert parity<=1e-5,('off parity',arm,parity);model.backbone.set_target_enabled(True);budget.count('gate_'+arm+'_on')
  with torch.inference_mode():on=logits(model,x).detach().cpu()
  valdataset=DATASETS.build(cfg.val_dataloader.dataset);begin=time.monotonic();vs=valdataset[0];io_seconds=time.monotonic()-begin;torch.cuda.synchronize();begin=time.monotonic();budget.count('gate_'+arm+'_development_timing')
  with torch.inference_mode():probe=model.test_step(pseudo_collate([vs]))[0]
  torch.cuda.synchronize();evaluation_seconds=time.monotonic()-begin;assert probe.pred_sem_seg.data.shape[-2:]==vs['data_samples'].gt_sem_seg.data.shape[-2:];gate_samples.append({'arm':arm,'one_development_model_seconds':evaluation_seconds,'one_input_load_seconds':io_seconds,'shape':list(probe.pred_sem_seg.data.shape)})
  if arm=='H':assert float((on-reference).abs().max())<=1e-5,'zero output head not identity'
  legacy_error=None
  if arm=='MH':
   legacy,_,_=build('MH',budget,legacy=True);legacy.load_state_dict(model.state_dict(),strict=True);legacy.eval();budget.count('gate_legacy_MH')
   with torch.inference_mode():prior=logits(legacy,x).detach().cpu()
   legacy_error=float((on-prior).abs().max());assert legacy_error<=1e-5;del legacy;gc.collect();torch.cuda.empty_cache()
  model.train();optimcfg=copy.deepcopy(cfg.optim_wrapper);optimcfg.constructor='Phase400PEFTConstructor';optim=build_optim_wrapper(model,optimcfg);sched=[LinearLR(optim,start_factor=.1,by_epoch=False,begin=0,end=100),PolyLR(optim,eta_min=0.,power=.9,by_epoch=False,begin=100,end=4000)];assert {id(p) for g in optim.optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
  data=model.data_preprocessor(batch,True);budget.count('gate_'+arm+'_loss');losses=model(**data,mode='loss');loss,_=model.parse_losses(losses);assert torch.isfinite(loss).item();optim.backward(loss);grads={n:{'finite':bool(torch.isfinite(p.grad).all()),'nonzero':int(torch.count_nonzero(p.grad))} for n,p in model.named_parameters() if p.requires_grad and p.grad is not None};assert grads and all(v['finite'] for v in grads.values()) and sum(v['nonzero'] for v in grads.values())>0;assert all(p.grad is None for p in model.parameters() if not p.requires_grad);optim.step();optim.zero_grad();budget.updates+=1
  for s in sched:s.step()
  now={n:tensor_sha(t) for n,t in model.state_dict().items() if n in frozen};assert now==frozen,'frozen parameter/buffer changed in one step'
  checkpoint={'state_dict':model.state_dict(),'optimizer':optim.state_dict(),'param_schedulers':[s.state_dict() for s in sched],'step':1,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'next_draw':1};cp=R/'gate_checkpoints'/(arm+'.pth');cp.parent.mkdir(exist_ok=True);torch.save(checkpoint,cp);saved=torch.load(cp,map_location='cpu');state_before=state_sha(model.state_dict());model.load_state_dict(saved['state_dict'],strict=True);optim.load_state_dict(saved['optimizer']);[s.load_state_dict(v) for s,v in zip(sched,saved['param_schedulers'])];assert state_sha(model.state_dict())==state_before and saved['step']==1 and saved['next_draw']==1;assert torch.equal(saved['torch_rng'],torch.get_rng_state());a=dataset[1,1];b=dataset[1,1];assert tensor_sha(a['inputs'])==tensor_sha(b['inputs']) and tensor_sha(a['data_samples'].gt_sem_seg.data)==tensor_sha(b['data_samples'].gt_sem_seg.data)
  reports.append({'arm':arm,'passed':True,'active':active,'names':names,'checkpoint_missing':load['missing'],'off_max_abs':parity,'initial_on_max_abs':float((on-reference).abs().max()),'legacy_MH_max_abs':legacy_error,'gradient_scope':grads,'frozen_parameters_and_buffers':True,'checkpoint_restore':True,'checkpoint_SHA':sha(cp),'build_and_gate_seconds':time.monotonic()-start,'initial':initial});del model,optim,sched,checkpoint,saved,data,losses,loss,x;gc.collect();torch.cuda.empty_cache();x=pre['inputs'];budget.check()
 assert initials['MH']['M']==initials['M']['M'] and initials['MH']['H']==initials['H']['H'];save('CORRECTNESS_GATE.json',{'passed':True,'reports':reports,'evaluation_probes':gate_samples,'independent_common_initializations':True,'no_default_MsRE_Source_equivalence_assumption':True,'old_sparse_unit_tests':3,'init_seeds':{'M':4000101,'H':4000102},'data_and_augment_seeds':{'order':40001,'per_draw_augment':4000103},'schedule':{'optimizer':js(R/'PREFLIGHT_READONLY.json')['resolved_65a']['optim_wrapper'],'schedulers':js(R/'PREFLIGHT_READONLY.json')['resolved_65a']['param_scheduler']}});budget.finish('passed')
if __name__=='__main__':
 try:main()
 except BaseException:
  save('GATE_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise
