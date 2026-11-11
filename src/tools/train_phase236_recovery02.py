"""One prescribed C-only trajectory, with no validation/test access or retries."""
from phase236_common import *
import argparse,copy,random,resource,sys,traceback

def main():
 global budget,out
 domain=sys.argv[1];seed=int(sys.argv[2]);lr=float(sys.argv[3]);assert domain in ['L8','SPARCS'] and seed in [21701,21702] and lr in [1e-4,3e-5]
 arm=f'{domain}_{seed}_{lr:.0e}';out=R/'fits'/arm;assert not out.exists();out.mkdir(parents=True)
 save(out/'STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'domain':domain,'seed':seed,'lr':lr})
 assert js(R/'FILL_GATE_ACCEPTED.json')['status']=='passed';assert sha(R/'code/phase236_valid_mask.py')==js(R/'FILL_GATE_ACCEPTED.json')['module_SHA']
 check_bundle();os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')]
 os.environ.update(XFORMERS_DISABLED='1',PHASE64_TARGET=domain.lower(),PHASE64_METHOD='shared_parent',PHASE64_PARENT_CHECKPOINT=str(SOURCE))
 import numpy as np,torch,rasterio
 from rasterio.windows import Window
 from PIL import Image
 from mmengine.config import Config
 from mmengine.dataset import InfiniteSampler,pseudo_collate
 from mmengine.registry import init_default_scope,PARAM_SCHEDULERS
 from mmengine.optim import build_optim_wrapper
 from mmseg.registry import DATASETS
 import cloud_adapter.datasets,cloud_adapter.models
 from phase236_valid_mask import install
 from unittest.mock import patch
 prepstart=time.monotonic();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True);init_default_scope('mmseg')
 train=js(R/'TRAIN_INPUTS.json')[domain];evals=js(R/'EVAL_INPUTS.json');forbidden={str(Path(v[k]).resolve()) for pools in evals.values() for rows in pools.values() for v in rows for k in ['image_path','mask_path']};denied=[]
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=str(Path(os.fsdecode(args[0])).resolve())
   if p in forbidden or ('cloudsen12_high_l1c' in p and Path(p).suffix.lower() in ['.png','.npz','.tif']):denied.append(p);raise RuntimeError('Forbidden training access: '+p)
 sys.addaudithook(audit)
 for v in train:
  assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256'],v['name']
 assert sha(SOURCE)==js(R/'LOCK.json')['Source_SHA256']
 cfg=Config.fromfile('configs/protocol/phase64_target_parent_rgb.py');cfg.randomness.seed=seed;cfg.optim_wrapper.optimizer.lr=lr
 assert cfg.train_cfg.max_iters==4000 and cfg.train_dataloader.batch_size==1
 dataset=DATASETS.build(cfg.train_dataloader.dataset);dataset.full_init();assert len(dataset)==len(train)
 bypath={str(Path(v['image_path']).resolve()):v for v in train};assert {str(Path(dataset.get_data_info(i)['img_path']).resolve()) for i in range(len(dataset))}==set(bypath)
 handles={};plan=js(R/'NATIVE_PLAN.json');bindings={v['name']:v for v in js(R/'TRAIN_BINDING235.json')}
 def native(v):
  if v['scene'] not in handles:handles[v['scene']]=rasterio.open(plan['train'][v['scene']]['path'])
  a=handles[v['scene']].read(1,window=Window(int(v['y']),int(v['x']),512,512),boundless=True,fill_value=0)
  assert set(np.unique(a))<={0,64,128,192,255};old=np.asarray(Image.open(v['mask_path']));lut=np.zeros(256,np.uint8);lut[64]=1;lut[192]=2;lut[255]=3;assert np.array_equal(lut[a],old)
  y=np.zeros(a.shape,np.uint8);y[a==64]=2;y[(a==192)|(a==255)]=1;y[a==0]=255
  b=bindings[v['name']];assert hashlib.sha256(a.tobytes()).hexdigest()==b['native_SHA'];assert hashlib.sha256((a!=0).tobytes()).hexdigest()==b['valid_SHA']
  return a,y
 if domain=='L8':
  for g,v in plan['train'].items():assert sha(v['path'])==v['SHA']
 census=[];active=[];support=collections.Counter()
 for i in range(len(dataset)):
  v=bypath[str(Path(dataset.get_data_info(i)['img_path']).resolve())]
  if domain=='L8':a,y=native(v)
  else:
   from evaluate_phase64_source_only_parent import SPARCS_TO_PARENT
   a=np.asarray(Image.open(v['mask_path']));y=SPARCS_TO_PARENT[a]
  hist={str(k):int((y==k).sum()) for k in [0,1,2,255]};support.update(hist)
  census.append({'index':i,'name':v['name'],'scene':v['scene'],'valid_pixels':int((y!=255).sum()),'support':hist,'label_SHA':hashlib.sha256(y.tobytes()).hexdigest()})
  if (y!=255).any():active.append(i)
 assert active
 if domain=='L8':
  oldanno=dataset.pipeline.transforms[1]
  def annotation(results):
   results=oldanno(results);v=bypath[str(Path(results['img_path']).resolve())];_,y=native(v);results['gt_seg_map']=y;return results
  dataset.pipeline.transforms[1]=annotation
 save(out/'TRAIN_POOL.json',{'census':census,'active_indices':active,'excluded_all_invalid':[v for v in census if v['index'] not in active],'scene_count':len({v['scene'] for v in train}),'valid_support':dict(support),'native_Fill_ignore_before_geometry':domain=='L8','config':cfg.to_dict(),'target_config_SHA':sha(SRC/'configs/protocol/phase64_target_parent_rgb.py')})
 sampler=iter(InfiniteSampler(active,shuffle=True,seed=seed))
 def augment(index,ds):
  oldnp=np.random.get_state();oldpy=random.getstate();np.random.seed(ds);random.seed(ds);draws=[]
  def wrap(name,fn):
   def f(*a,**kw):
    v=fn(*a,**kw);draws.append({'function':name,'args':a,'kwargs':kw,'value':v.tolist() if hasattr(v,'tolist') else v});return v
   return f
  try:
   with patch.object(np.random,'randint',wrap('randint',np.random.randint)),patch.object(np.random,'uniform',wrap('uniform',np.random.uniform)),patch.object(np.random,'choice',wrap('choice',np.random.choice)):
    data=dataset.pipeline(dataset.get_data_info(index))
   valid=data['data_samples'].gt_sem_seg.data!=255
   proof={'image_SHA':tsha(data['inputs']),'label_SHA':tsha(data['data_samples'].gt_sem_seg.data),'valid_SHA':tsha(valid),'draws':draws,'flip':data['data_samples'].metainfo.get('flip'),'flip_direction':data['data_samples'].metainfo.get('flip_direction')}
   return data,proof
  finally:np.random.set_state(oldnp);random.setstate(oldpy)
 # No model forward until all input and Fill census checks have passed.
 save(out/'PREPARATION.json',{'seconds':time.monotonic()-prepstart,'all_original_input_SHA_passed':True,'train_only':True})
 budget=Budget('train_recovery02_'+arm);wrapper,_=build(budget);model=wrapper.segmentor
 ck=torch.load(SOURCE,map_location='cpu')['state_dict'];states=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,states[n].detach().cpu()) for n,t in ck.items());del ck
 frozen={n:t.detach().cpu().clone() for n,t in states.items() if not n.startswith('decode_head.')};initial=digest(model.decode_head.state_dict());patchinfo=install(model.decode_head) if domain=='L8' else None
 model.train()
 for n,m in model.named_modules():
  if n and not n.startswith('decode_head'):m.eval()
 model.decode_head.train()
 for n,p in model.named_parameters():p.requires_grad=n.startswith('decode_head.')
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1316516
 opt=build_optim_wrapper(model,copy.deepcopy(cfg.optim_wrapper));schedulers=[PARAM_SCHEDULERS.build(copy.deepcopy(c),default_args={'optimizer':opt}) for c in cfg.param_scheduler]
 save(out/'INITIAL.json',{'head_digest':initial,'Source_SHA':sha(SOURCE),'nonhead_digest':digest(frozen),'optimizer_groups':[{'lr':g['lr'],'weight_decay':g['weight_decay'],'parameters':sum(p.numel() for p in g['params'])} for g in opt.optimizer.param_groups],'optim_config':cfg.optim_wrapper,'schedulers':cfg.param_scheduler,'trainable':1316516,'all_nonhead_Source_exact':True})
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
 losses=(out/'LOSSES.jsonl').open('x',buffering=1);schedule=(out/'SCHEDULE.jsonl').open('x',buffering=1);checks=[];exposure=collections.Counter();backwards=0;updates=0;opt.zero_grad()
 def frozen_check(step):
  now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items());assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('decode_head.'));assert not model.backbone.training
  checks.append({'step':step,'nonhead_exact':True,'nonhead_no_gradient':True,'nonhead_digest':digest(frozen)})
 try:
  for step in range(1,4001):
   index=active[next(sampler)];v=bypath[str(Path(dataset.get_data_info(index)['img_path']).resolve())];ds=seed*10000+step-1;data,proof=augment(index,ds);exposure[v['scene']]+=1
   if domain=='L8':model.decode_head._phase236_generator.manual_seed(seed*1000000+step);model.decode_head._phase236_sampling=[]
   budget.current={'arm':arm,'step':step,'input':v['name']};batch=model.data_preprocessor(pseudo_collate([data]),True);budget.count('training_full_input')
   terms=model(**batch,mode='loss');loss,logs=model.parse_losses(terms);assert torch.isfinite(loss)
   opt.backward(loss);backwards+=1;grads=[p.grad for p in model.decode_head.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
   opt.step();updates+=1;lr_used=[g['lr'] for g in opt.optimizer.param_groups];opt.zero_grad()
   for scheduler in schedulers:scheduler.step()
   rec={'step':step,'index':index,'name':v['name'],'scene':v['scene'],'data_seed':ds,**proof};schedule.write(json.dumps(rec,default=str)+'\n')
   losses.write(json.dumps({'step':step,'loss':float(loss.detach()),'terms':{k:float(z.detach()) for k,z in logs.items()},'lr_used':lr_used,'sampling_checks':getattr(model.decode_head,'_phase236_sampling',[])})+'\n')
   if step%100==0:frozen_check(step);budget.check();save(out/'PROGRESS.json',{'step':step,'updates':updates,'backwards':backwards,'scene_exposure':dict(exposure)})
   if step in [500,1000,2000,4000]:
    head={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};hp=out/f'head_{step}.pth';torch.save({'state_dict':head,'step':step,'seed':seed,'lr':lr,'Source_SHA':sha(SOURCE)},hp)
    rp=out/f'resume_{step}.pth';torch.save({'head':head,'optimizer':opt.state_dict(),'schedulers':[s.state_dict() for s in schedulers],'torch_RNG':torch.get_rng_state(),'cuda_RNG':torch.cuda.get_rng_state_all(),'numpy_RNG':np.random.get_state(),'python_RNG':random.getstate(),'valid_sampling_RNG':model.decode_head._phase236_generator.get_state() if domain=='L8' else None,'step':step,'sampler_prefix':step,'next_data_seed':seed*10000+step,'base_seed':seed},rp)
    save(out/f'SAVE_{step}.json',{'step':step,'head_SHA':sha(hp),'head_digest':digest(head),'resume_SHA':sha(rp),'nonhead_digest':digest(frozen),'restore_sampler':'rebuild original InfiniteSampler seed then advance saved prefix; per-step data seeds fixed'})
   del loss,terms,grads,batch,data
  losses.close();schedule.close();frozen_check(4000)
  partner=R/'fits'/f'{domain}_{seed}_{1e-4:.0e}'
  pair=None
  if lr==3e-5:
   assert sha(out/'SCHEDULE.jsonl')==sha(partner/'SCHEDULE.jsonl'),'LR paired input/augmentation drift';assert initial==js(partner/'INITIAL.json')['head_digest'];pair={'schedule_SHA':sha(out/'SCHEDULE.jsonl'),'other_arm':partner.name,'exact':True}
  save(out/'COMPLETE.json',{'domain':domain,'seed':seed,'lr':lr,'updates':updates,'backwards':backwards,'schedule_SHA':sha(out/'SCHEDULE.jsonl'),'losses_SHA':sha(out/'LOSSES.jsonl'),'scene_exposure':dict(exposure),'nonhead_checks':checks,'paired_LR':pair,'forbidden_accesses':denied,'heads':{str(s):js(out/f'SAVE_{s}.json') for s in [500,1000,2000,4000]}})
  budget.finish('passed');(out/'EXIT_CODE').write_text('0\n')
 finally:
  for h in handles.values():h.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  if 'out' in globals():save(out/'FAILURE.json',{'traceback':traceback.format_exc()});(out/'EXIT_CODE').write_text('1\n')
  if 'budget' in globals():budget.finish('failed')
  raise
