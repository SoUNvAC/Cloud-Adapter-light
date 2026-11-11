"""48 fixed Source/C gradient forwards plus16 A mask invariance checks, no updates."""
from phase240_common import *
from phase240_gradients import paired_gradients
import sys, traceback


def main():
 global budget
 check_bundle();assert not (R/'DIAGNOSTICS_STARTED.json').exists();save('DIAGNOSTICS_STARTED.json',{'PID':os.getpid()})
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch,rasterio
 from rasterio.windows import Window
 from mmengine.structures import PixelData
 from mmseg.structures import SegDataSample
 from mmengine.dataset import pseudo_collate
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 from phase236_valid_mask import install
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
 allowed=js(R/'DIAGNOSTIC_INPUTS.json');evals=js(R/'EVAL_INPUTS.json')
 forbidden={str(Path(v[k]).resolve()) for pools in evals.values() for rows in pools.values() for v in rows for k in ['image_path','mask_path']}
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=str(Path(os.fsdecode(args[0])).resolve())
   if p in forbidden or ('cloudsen12_high_l1c' in p and Path(p).suffix.lower() in ['.png','.npz','.tif']):raise RuntimeError('Diagnostic forbidden access '+p)
 sys.addaudithook(audit)
 budget=Budget('diagnostics');wrapper,_=build(budget);model=wrapper.segmentor.eval();source={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()}
 plans=js(R/'NATIVE_PLAN.json');bindings={v['name']:v for v in js(R/'TRAIN_BINDING235.json')};rows=[];gaps=[];handles={}
 for domain in ['L8','SPARCS']:
  if domain=='L8':install(model.decode_head)
  else:
   # Avoid carrying L8's sampling closures into SPARCS: fresh process-local model.
   # The all-valid path is identical, but clear active state before each batch.
   model.decode_head._phase236_active_valid=None
  states=[('Source',None),('C2000',OLD/f'fits/{domain}_21701_1e-04/head_2000.pth'),('C4000',OLD/f'fits/{domain}_21701_1e-04/head_4000.pth')]
  ap=R/'fits'/f'{domain}_A'/'head_4000.pth'
  if ap.exists():states.append(('A4000',ap))
  else:gaps.append({'domain':domain,'state':'A4000','reason':'missing_endpoint_no_refit'})
  source_masks={};source_cls={}
  for state,path in states:
   if path is not None and not path.exists():gaps.append({'domain':domain,'state':state,'reason':'missing_existing_checkpoint_no_refit'});continue
   if path is not None:
    if state.startswith('C'):
     identity=next(z for z in js(R/'C_HEAD_IDENTITIES.json') if z['fit']==f'{domain}_21701_1e-04' and z['step']==int(state[1:]));assert sha(path)==identity['head_SHA']
    else:assert sha(path)==js(path.parent/'SAVE_4000.json')['head_SHA']
   model.decode_head.load_state_dict(source if path is None else torch.load(path,map_location='cpu')['state_dict']);model.eval()
   for n,p in model.named_parameters():p.requires_grad=n.startswith('decode_head.') and state!='A4000'
   params=[p for p in model.decode_head.parameters() if p.requires_grad]
   capture={}
   def hook(m,args,outputs):
    capture['mask_SHA']=digest({str(i):t for i,t in enumerate(outputs[1])});capture['class_SHA']=digest({str(i):t for i,t in enumerate(outputs[0])})
   token=model.decode_head.register_forward_hook(hook)
   for index,v in enumerate(allowed[domain]):
    assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
    if domain=='L8':
     p=plans['train'][v['scene']]
     if v['scene'] not in handles:handles[v['scene']]=rasterio.open(p['path'])
     a=handles[v['scene']].read(1,window=Window(int(v['y']),int(v['x']),512,512),boundless=True,fill_value=0);b=bindings[v['name']];assert hashlib.sha256(a.tobytes()).hexdigest()==b['native_SHA']
     y=np.zeros(a.shape,np.uint8);y[a==64]=2;y[(a==192)|(a==255)]=1;y[a==0]=255
    else:y=load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8)
    x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)[0]).to(torch.uint8)
    sample=SegDataSample();sample.set_metainfo(dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512),padding_size=[0,0,0,0]));sample.gt_sem_seg=PixelData(data=torch.from_numpy(y[None].astype(np.int64)))
    batch=model.data_preprocessor(pseudo_collate([dict(inputs=x,data_samples=sample)]),True)
    if hasattr(model.decode_head,'_phase236_generator'):model.decode_head._phase236_generator.manual_seed(240000+index)
    torch.manual_seed(240000+index);torch.cuda.manual_seed_all(240000+index)
    budget.current={'domain':domain,'state':state,'input':v['name']};budget.count('diagnostic_full_input')
    if state=='A4000':
     with torch.no_grad():terms=model(**batch,mode='loss')
     stats=None;assert capture['mask_SHA']==source_masks[v['name']]
    else:
     terms=model(**batch,mode='loss');loss,_=model.parse_losses(terms);grads,stats,_=paired_gradients(loss,terms,params,budget);del grads,loss
     if state=='Source':source_masks[v['name']]=capture['mask_SHA'];source_cls[v['name']]=capture['class_SHA']
    rows.append(dict(domain=domain,state=state,name=v['name'],scene=v['scene'],mask_SHA=capture['mask_SHA'],class_SHA=capture['class_SHA'],gradient=stats,
                     A_mask_Source_exact=capture['mask_SHA']==source_masks.get(v['name']) if state=='A4000' else None,
                     A_class_Source_changed=capture['class_SHA']!=source_cls.get(v['name']) if state=='A4000' else None))
    del terms,batch
   token.remove()
 save('DIAGNOSTICS_COMPLETE.json',{'rows':rows,'gaps':gaps,'optimizer_updates':0,'no_features_archived':True,'geometry':'unaugmented_fixed512_evaluation_geometry'})
 budget.finish('passed')
 for h in handles.values():h.close()

if __name__=='__main__':
 try:main()
 except BaseException:
  save('DIAGNOSTICS_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise
