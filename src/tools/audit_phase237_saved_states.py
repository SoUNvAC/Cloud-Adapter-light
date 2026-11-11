"""Read-only CPU inspection of retained states and runtime dependency identities.
No model is constructed or called; weights remain on the training host.
"""
from pathlib import Path
import hashlib,importlib,inspect,json,os,resource,sys,time
R=Path('/home/scv/shared/phase237_20261028');SRC=Path('/home/scv/Cloud-Adapter-light/src')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def main():
 start=time.monotonic();assert (R/'EXIT_CODE').read_text().strip()=='0';assert not (R/'SAVED_STATE_CPU_AUDIT.json').exists();sys.path[:0]=[str(SRC),str(SRC/'tools')];os.chdir(SRC)
 import torch,numpy as np
 torch.set_num_threads(2)
 versions={};dependencies={}
 for name in ['torch','numpy','mmcv','mmseg','mmdet','mmengine','mmcv.ops.point_sample','mmdet.models.dense_heads.mask2former_head','mmdet.models.dense_heads.maskformer_head','mmseg.models.decode_heads.mask2former_head','mmdet.models.task_modules.assigners.hungarian_assigner','mmdet.models.task_modules.assigners.match_cost','mmengine.optim.scheduler.param_scheduler','mmengine.optim.optimizer.optimizer_wrapper','mmengine.optim.optimizer.default_constructor','mmseg.datasets.transforms.transforms','mmseg.datasets.transforms.loading','cloud_adapter.datasets','export_phase12_onnx','validate_phase12_onnxruntime']:
  mod=importlib.import_module(name);p=Path(inspect.getfile(mod));dependencies[name]={'path':str(p),'SHA':sha(p),'bytes':p.stat().st_size};versions[name]=getattr(mod,'__version__',None)
 source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)==js(R/'LOCK.json')['Source_SHA256']
 records=[]
 for p in sorted((R/'fits').glob('*/COMPLETE.json')):
  complete=js(p)
  for step in [500,1000,2000,4000]:
   hp=p.parent/f'head_{step}.pth';rp=p.parent/f'resume_{step}.pth';receipt=js(p.parent/f'SAVE_{step}.json');assert sha(hp)==receipt['head_SHA'] and sha(rp)==receipt['resume_SHA']
   head=torch.load(hp,map_location='cpu');resume=torch.load(rp,map_location='cpu');assert head['step']==resume['step']==step and head['seed']==complete['seed'] and head['lr']==complete['lr'];assert set(head['state_dict'])==set(resume['head']);assert all(torch.equal(t,resume['head'][n]) and torch.isfinite(t).all() for n,t in head['state_dict'].items());assert sum(t.numel() for t in head['state_dict'].values())==1316516
   assert resume['base_seed']==complete['seed'] and resume['next_data_seed']==complete['seed']*10000+step and resume['sampler_prefix']==step
   assert resume['torch_RNG'].dtype==torch.uint8 and resume['cuda_RNG'] and all(t.dtype==torch.uint8 for t in resume['cuda_RNG']);assert resume['numpy_RNG'][0]=='MT19937' and resume['python_RNG']
   assert (resume['valid_sampling_RNG'] is not None)==(complete['domain']=='L8')
   opt=resume['optimizer'];opt=opt.get('optimizer',opt);assert opt['state'] and opt['param_groups'];states=list(opt['state'].values());assert all(float(s['step'])==step and torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() for s in states);assert len(resume['schedulers'])==2
   records.append({'fit':p.parent.name,'step':step,'head_SHA':receipt['head_SHA'],'resume_SHA':receipt['resume_SHA'],'head_tensors':len(head['state_dict']),'head_parameters':1316516,'head_resume_tensors_exact':True,'optimizer_tensors_finite_and_step_exact':True,'optimizer_state_count':len(states),'scheduler_states':resume['schedulers'],'all_training_RNG_and_sampler_restoration_fields_present':True})
   del head,resume,opt,states
   assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3
 assert len(records)==24
 out={'status':'passed','neural_model_calls':0,'training_updates':0,'CPU_only':True,'versions':versions,'dependencies_SHA':dependencies,'source_SHA':sha(source),'records':records,'CPU_seconds':time.monotonic()-start,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'dependency_audit_timing':'post-run read-only snapshot; not claimed pre-sealed dependency binary hashes'}
 (R/'SAVED_STATE_CPU_AUDIT.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out))
if __name__=='__main__':main()
