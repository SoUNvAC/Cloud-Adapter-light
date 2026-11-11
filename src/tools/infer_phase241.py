"""One fixed pass, full FP32 cache; old236 prediction and support are hard gates."""
from phase241_common import *
import sys,traceback

def main():
 global budget
 stage=sys.argv[1];assert stage in ('val','test');check_bundle()
 assert not (R/f'{stage.upper()}_STARTED.json').exists(),'No blind inference restart'
 if stage=='test':
  lock=js(R/'ALIGNMENT_SEAL.json')
  for n,h in lock['files'].items():assert sha(R/n)==h
  assert js(R/'VAL_COMPLETE.json')['passed']
 save(stage.upper()+'_STARTED.json',{'epoch':time.time(),'PID':os.getpid()})
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch,rasterio
 from rasterio.windows import Window
 from PIL import Image
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 identities=js(R/'MODEL_IDENTITIES.json');assert sha(SOURCE)==identities['Source_SHA256']
 budget=Budget('infer_'+stage);wrapper,_=build(budget);model=wrapper.segmentor.eval();model.requires_grad_(False)
 initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()}
 frozen={n:t.detach().cpu().clone() for n,t in dict(list(model.named_parameters())+list(model.named_buffers())).items() if not n.startswith('decode_head.')}
 save(stage.upper()+'_BACKEND.json',{'float':'float32','cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'nonhead_digest':digest(frozen),'Source_head_digest':digest(initial)})
 E=js(R/'EVAL_INPUTS.json');T=js(R/'TRAIN_INPUTS.json');plan=js(R/'NATIVE_PLAN.json');valrefs={v['name']:v for v in js(R/'L8_VAL_NATIVE_REFERENCES.json')};expected={(v['split'],v['name']):v for v in js(R/'EXPECTED_NATIVE234.json')};handles={}
 denied={str(Path(v[k]).resolve()) for rows in T.values() for v in rows for k in ('image_path','mask_path')}
 if stage=='val':denied|={str(Path(v[k]).resolve()) for pools in E.values() for v in pools['test'] for k in ('image_path','mask_path')}
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=str(Path(os.fsdecode(args[0])).resolve())
   if p in denied or ('cloudsen12_high_l1c' in p and Path(p).suffix.lower() in ('.png','.npz','.tif')):raise RuntimeError('241 access denied '+p)
 sys.addaudithook(audit)
 old={}
 for v in [json.loads(l) for l in (R/'EXPECTED236.jsonl').read_text().splitlines()]:
  k=(v['domain'],v['split'],v['name'],v['model']);assert k not in old;old[k]=v
 def label(v,domain):
  if domain=='SPARCS':return load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8)
  if stage=='val':
   r=valrefs[v['name']];assert sha(r['file'])==r['sha256']
   with np.load(r['file'],allow_pickle=False) as z:a=z['native'];valid=z['valid'];truth=z['truth']
  else:
   p=plan['test'][v['scene']]
   if v['scene'] not in handles:assert sha(p['path'])==p['SHA'];handles[v['scene']]=rasterio.open(p['path'])
   a=handles[v['scene']].read(1,window=Window(int(v['y']),int(v['x']),512,512),boundless=True,fill_value=0);valid=a!=0;truth=np.zeros(a.shape,np.uint8);truth[a==64]=2;truth[(a==192)|(a==255)]=1;truth[~valid]=255
  lut=np.zeros(256,np.uint8);lut[64]=1;lut[192]=2;lut[255]=3;assert np.array_equal(lut[a],np.asarray(Image.open(v['mask_path'])))
  ex=expected[stage,v['name']];assert hashlib.sha256(a.tobytes()).hexdigest()==ex['native_array_SHA'];assert hashlib.sha256(valid.tobytes()).hexdigest()==ex['valid_array_SHA'];return truth
 def npy(path,a):
  path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();tmp=path.with_suffix('.tmp')
  with tmp.open('xb') as f:np.save(f,a,allow_pickle=False)
  tmp.replace(path);return sha(path)
 index=(R/f'{stage.upper()}_SCORE_INDEX.jsonl').open('x',buffering=1);repeats=[];count=0
 for domain in ('L8','SPARCS'):
  heads={'Source':initial}
  for v in identities['C']:
   if v['domain']==domain:
    assert sha(v['remote_path'])==v['head_SHA256'];heads['C'+str(v['seed'])]=torch.load(v['remote_path'],map_location='cpu')['state_dict']
  for i,v in enumerate(E[domain][stage]):
   assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256'];y=label(v,domain)
   lp=R/'labels'/stage/domain/(v['name']+'.npy');lh=npy(lp,y)
   x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda()
   for name,head in heads.items():
    model.decode_head.load_state_dict(head);budget.current={'domain':domain,'split':stage,'input':i,'model':name,'name':v['name']};budget.count('full_input')
    with torch.inference_mode():p=wrapper(x).float().clamp_min(0);p=p/p.sum(1,keepdim=True).clamp_min(1e-12)
    a=p[0].cpu().numpy();assert a.dtype==np.float32 and a.shape==(3,512,512) and np.isfinite(a).all()
    fp=R/'scores'/stage/domain/name/(v['name']+'.npy');fh=npy(fp,a);pred=a.argmax(0).astype(np.uint8);valid=y!=255;cm=np.bincount((3*y[valid].astype(np.int64)+pred[valid]).ravel(),minlength=9).reshape(3,3)
    receipt={'domain':domain,'split':stage,'name':v['name'],'scene':v['scene'],'usgs_shadows':v.get('usgs_shadows'),'model':name,'score_file':str(fp.relative_to(R)),'score_SHA':fh,'label_file':str(lp.relative_to(R)),'label_SHA':lh,'prediction_SHA':hashlib.sha256(pred.tobytes()).hexdigest(),'confusion':cm.tolist()}
    ex=old[domain,stage,v['name'],name];assert receipt['prediction_SHA']==ex['prediction_SHA'] and receipt['confusion']==ex['confusion'],('236 reproduction',receipt,ex)
    receipt['old236_exact']=True;index.write(json.dumps(receipt)+'\n');count+=1
    if stage=='val' and i==0:
     budget.count('repeat_first_val')
     with torch.inference_mode():q=wrapper(x).float().clamp_min(0);q=q/q.sum(1,keepdim=True).clamp_min(1e-12)
     b=q[0].cpu().numpy();rp=R/'repeats'/domain/(name+'.npy');rh=npy(rp,b);repeats.append({'domain':domain,'model':name,'first_val_input':v['name'],'file':str(rp.relative_to(R)),'SHA':rh,'max_abs_difference':float(np.max(np.abs(a-b))),'argmax_disagreements':int(np.count_nonzero(a.argmax(0)!=b.argmax(0)))});del q,b
    del p,a,pred
   del x
   if i%50==0:budget.check()
  states=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,states[n].detach().cpu()) for n,t in frozen.items())
 index.close();assert count==(2919 if stage=='val' else 6531),count
 save(stage.upper()+'_COMPLETE.json',{'passed':True,'epoch':time.time(),'inputs_models':count,'score_index_SHA':sha(R/f'{stage.upper()}_SCORE_INDEX.jsonl'),'full_FP32_saved':True,'old236_exact':True,'nonhead_Source':True,'repeats':repeats});budget.finish('passed')
 for h in handles.values():h.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  save(sys.argv[1].upper()+'_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'budget' in globals():budget.finish('failed')
  raise
