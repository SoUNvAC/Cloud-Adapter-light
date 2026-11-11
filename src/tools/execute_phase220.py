"""Frozen shared-feature inference only. Fixed val joint counts and replay weight reproduction."""
from pathlib import Path
import argparse,collections,copy,hashlib,json,os,resource,sys,threading,time,traceback
R=Path('/home/scv/shared/phase220_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def main(mode):
 out=R/mode;out.mkdir(exist_ok=False);os.chdir(SRC);sys.path[:0]=[str(R/'code'),str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 import torch.nn.functional as F
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SOURCE_TO_PARENT
 from phase219_weighting import build,rawsha
 from controller_phase220 import host
 assert np.__version__=='1.26.4';torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=(mode=='replay');torch.use_deterministic_algorithms(mode=='replay',warn_only=mode=='replay')
 prior=js(R/'BUDGET_STATE.json');start=time.monotonic();counts=collections.Counter();index=0;done=threading.Event();samples=[];denied=[]
 ledger=(out/'CALL_LEDGER.jsonl').open('x',buffering=1)
 def status():save(out/'RUN_STATUS.json',{'PID':os.getpid(),'pool':mode,'input':index,'new_units':sum(counts.values()),'cumulative_units':prior['units']+sum(counts.values()),'GPU_seconds':time.monotonic()-start,'counts':dict(counts)})
 def count(kind):
  assert prior['units']+sum(counts.values())<6000 and prior['GPU_seconds']+time.monotonic()-start<3600;counts[kind]+=1;ledger.write(json.dumps({'ordinal':prior['units']+sum(counts.values()),'pool':mode,'input':index,'kind':kind})+'\n')
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.path.abspath(os.fsdecode(args[0])).replace('\\','/')
   if 'cloudsen12_high_l1c/ann_dir/train/' in p or 'cloudsen12_high_l1c/ann_dir/test/' in p or 'cloudsen12_high_l1c/img_dir/test/' in p or (mode=='replay' and 'cloudsen12_high_l1c/ann_dir/' in p) or '/data/sparcs' in p.lower() or '/data/l8_biome' in p.lower():
    denied.append(p);raise RuntimeError('220 forbidden data access: '+p)
 sys.addaudithook(audit)
 def guard():
  while not done.wait(15):
   try:
    cpu=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0;reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0;size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    assert cpu<16*1024**3 and max(allocated,reserved)<20*1024**3 and size<2_000_000_000 and prior['GPU_seconds']+time.monotonic()-start<3600
    samples.append({'CPU_peak_bytes':cpu,'GPU_allocated':allocated,'GPU_reserved':reserved,'bytes':size});save(out/'RESOURCE_SAMPLES.json',samples);host();status()
   except BaseException:
    (out/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(out/'EXIT_CODE').write_text('71\n');os._exit(71)
 threading.Thread(target=guard,daemon=True).start()
 try:
  lock=js(R/'MODEL_LOCK.json');assert sha(lock['Source']['path'])==lock['Source']['SHA'];wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=lock['Source']['path'],precision='fp32',active_block_indices=None));model=wrapper.segmentor.eval()
  for p in model.parameters():p.requires_grad=False
  actual=dict(list(model.named_parameters())+list(model.named_buffers()));source_ck=torch.load(lock['Source']['path'],map_location='cpu')['state_dict'];assert all(torch.equal(v,actual[n].detach().cpu()) for n,v in source_ck.items())
  original={n:v.detach().cpu().clone() for n,v in actual.items()};teacher=copy.deepcopy(model.decode_head).eval()
  for p in teacher.parameters():p.requires_grad=False
  initial={n:v.detach().cpu().clone() for n,v in teacher.state_dict().items()};heads={}
  if mode=='val':
   for key,v in lock['students'].items():
    assert sha(v['path'])==v['SHA'];heads[key]=torch.load(v['path'],map_location='cpu')['state_dict'];assert set(heads[key])==set(initial)
  def frozen():
   now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(v,now[n].detach().cpu()) for n,v in original.items() if not n.startswith('decode_head.'));assert all(torch.equal(v,teacher.state_dict()[n].detach().cpu()) for n,v in initial.items());assert all(p.grad is None and not p.requires_grad for p in model.parameters()) and all(p.grad is None and not p.requires_grad for p in teacher.parameters())
  def semantic(head,features,kind):
   count(kind);cl,ma=head(features,None);return torch.einsum('bqc,bqhw->bchw',F.softmax(cl[-1].float(),-1)[...,:-1],F.interpolate(ma[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid())
  def prediction(z):
   z=z.float().clamp_min(0);return (z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
  stream=(out/('JOINT_COUNTS.jsonl' if mode=='val' else 'REPLAY_WEIGHTS.jsonl')).open('x',buffering=1);weightstream=(out/'VAL_WEIGHT_EXPOSURE.jsonl').open('x',buffering=1) if mode=='val' else None;predhash={k:hashlib.sha256() for k in heads};teacherhash=hashlib.sha256();metadata=js(R/'SUPPORT_LOCK.json');expected=js(R/'EXPECTED_VAL.json') if mode=='val' else {v['sample_id']:v for v in js(R/'WEIGHT_MANIFEST.json')};replayed=[]
  with torch.inference_mode():
   for index,row in enumerate(metadata[mode],1):
    assert sha(row['image_path'])==row['image_SHA'];x=torch.from_numpy(load_rgb_image(Path(row['image_path']),512)).cuda();count(mode+'_shared_backbone');features=model.extract_feat(wrapper.normalize(x));ts=semantic(teacher,features,mode+'_teacher_head');tp=prediction(ts);conf=F.log_softmax(ts,1).exp().max(1).values[0].cpu().numpy();before=np.random.get_state();g,s,proof=build(conf,row['sample_id']);after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:];high=g.reshape(512,512)==1.5;shigh=s.reshape(512,512)==1.5;teacherhash.update(tp.tobytes());frozen()
    if mode=='replay':
     e=expected[row['sample_id']]
     for k in ['confidence_raw_SHA','G_SHA','S_SHA','permutation_SHA','stable_rank_SHA']:assert proof[k]==e[k],(row['sample_id'],k,proof[k],e[k])
     fields=[]
     for c in range(3):
      mask=tp==c;n=int(mask.sum());gh=int((mask&high).sum());ss=int((mask&shigh).sum());fields.append({'teacher_prediction_class':c,'support':n,'G_high_pixels':gh,'S_high_pixels':ss,'G_weight_sum':.5*n+gh,'S_weight_sum':.5*n+ss})
     replayed.append(row['sample_id']);stream.write(json.dumps({'sample_id':row['sample_id'],'name':row['name'],'image_SHA':row['image_SHA'],'teacher_prediction_SHA':rawsha(tp),'repeat_count':row['repeat_count'],'geometry_pixels':tp.size,'class_exposure':fields,'weight_proof':proof})+'\n')
    else:
     assert sha(row['mask_path'])==row['mask_SHA'];y=load_parent_target(Path(row['mask_path']),SOURCE_TO_PARENT,tp.shape);valid=y!=255;e=expected['per_input'][row['name']];assert rawsha(tp)==e['teacher_prediction_SHA'] and rawsha(high)==e['high_mask_SHA'];students={};studentsha={};normalfirst={}
     # Gate calls on the first fixed input are counted; its regular heads/features are reused.
     if index==1:
      cloned=copy.deepcopy(teacher).eval();cs=semantic(cloned,features,'gate_cloned_teacher_head');assert torch.equal(cs,ts);assert all(torch.equal(v,cloned.state_dict()[n].detach().cpu()) for n,v in initial.items());count('gate_official_Source_full');assert torch.equal(wrapper(x).float(),ts);del cloned
     for key,h in heads.items():
      model.decode_head.load_state_dict(h);model.eval();z=semantic(model.decode_head,features,'val_student_head');pred=prediction(z);students[key]=pred;studentsha[key]=rawsha(pred);predhash[key].update(pred.tobytes());cm=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3);assert cm.tolist()==e['students'][key]['confusion']
      if 'prediction_SHA' in e['students'][key]:assert studentsha[key]==e['students'][key]['prediction_SHA']
      if index==1:
       count('gate_official_student_full');official=wrapper(x).float();assert torch.equal(official,z) and np.array_equal(prediction(official),pred);normalfirst[key]={'raw_semantic_equal':True,'pred_SHA':studentsha[key]}
      assert all(torch.equal(v,model.decode_head.state_dict()[n].detach().cpu()) for n,v in h.items());frozen()
     if index==1:save(out/'GATE_ACCEPTED.json',{'first_input':row['name'],'zero_updates':True,'cloned_teacher_equal':True,'official_Source_full_equal':True,'all6_official_student_paths_equal':normalfirst,'teacher_prediction_SHA':rawsha(tp),'high_mask_SHA':rawsha(high),'weights':proof,'gradients_absent':True,'flags':{'deterministic':False,'cudnn_deterministic':False,'cudnn_benchmark':False},'counts':dict(counts)})
     for seed in [21701,21702]:
      d,gg,ss=[students[f'{seed}/{a}'] for a in ['D','G','S']];code=(~high[valid]).astype(np.int64)*243+y[valid]*81+tp[valid].astype(np.int64)*27+d[valid].astype(np.int64)*9+gg[valid].astype(np.int64)*3+ss[valid].astype(np.int64);j=np.bincount(code,minlength=486).reshape(2,3,3,3,3,3)
      for ai,a in enumerate(['D','G','S']):
       axes=tuple(k for k in range(6) if k not in [1,2,3+ai]);assert j.sum(axis=axes).tolist()==e['students'][f'{seed}/{a}']['reference_teacher_student_counts']
      for bi,band in enumerate(['high','low']):assert j[bi].sum(axis=(2,3,4)).tolist()==e['bands'][band]['confusion']
      stream.write(json.dumps({'seed':seed,'name':row['name'],'ROI':row['ROI'],'sample_id':row['sample_id'],'image_SHA':row['image_SHA'],'mask_SHA':row['mask_SHA'],'valid_pixels':int(valid.sum()),'ignore_pixels':int((~valid).sum()),'teacher_prediction_SHA':rawsha(tp),'student_prediction_SHA':{a:studentsha[f'{seed}/{a}'] for a in ['D','G','S']},'high_mask_SHA':rawsha(high),'S_mask_SHA':rawsha(shigh),'shape':[2,3,3,3,3,3],'J':j.tolist()})+'\n')
     exposure=[]
     for basis,category,validmask in [('reference_valid',y,valid),('teacher_prediction_valid',tp,valid),('teacher_prediction_geometry',tp,np.ones(tp.shape,bool))]:
      for c in range(3):
       v=(category==c)&validmask;n=int(v.sum());gh=int((v&high).sum());ss=int((v&shigh).sum());exposure.append({'basis':basis,'class':c,'support':n,'G_high_pixels':gh,'S_high_pixels':ss,'G_weight_sum':.5*n+gh,'S_weight_sum':.5*n+ss})
     weightstream.write(json.dumps({'name':row['name'],'ROI':row['ROI'],'sample_id':row['sample_id'],'high_mask_SHA':rawsha(high),'S_high_mask_SHA':rawsha(shigh),'S_permutation_SHA':proof['permutation_SHA'],'weight_exposure':exposure,'meaning':'val diagnostic reconstruction, not replay-training gradients'})+'\n')
    if index%25==0:status()
  stream.close()
  if weightstream:weightstream.close()
  if mode=='val':
   assert counts=={'val_shared_backbone':535,'val_teacher_head':535,'val_student_head':3210,'gate_cloned_teacher_head':1,'gate_official_Source_full':1,'gate_official_student_full':6};assert teacherhash.hexdigest()==expected['teacher_prediction_SHA'];assert all(v.hexdigest()==expected['student_prediction_SHA'][k] for k,v in predhash.items())
  else:assert counts=={'replay_shared_backbone':256,'replay_teacher_head':256} and len(replayed)==256
  frozen();done.set();save(out/'ACCESS_AUDIT.json',{'forbidden_attempts':denied,'source_train_label_reads':0,'official_source_test_reads':0,'target_inference':0,'training_steps':0,'backwards':0,'fits':0});save(out/'COMPLETE.json',{'new_units':sum(counts.values()),'counts':dict(counts),'GPU_seconds':time.monotonic()-start,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),'all_frozen_tensors_checked_each_input':True,'no_backward_fit_training':True,'teacher_prediction_SHA':teacherhash.hexdigest(),'student_prediction_SHA':{k:v.hexdigest() for k,v in predhash.items()},'numpy':np.__version__,'flags':{'deterministic_algorithms':mode=='replay','warn_only':mode=='replay','cudnn_deterministic':mode=='replay','cudnn_benchmark':False,'precision':'FP32'}});(out/'EXIT_CODE').write_text('0\n');status()
 except BaseException:
  done.set();(out/'FAILURE.txt').write_text(traceback.format_exc());save(out/'FAILURE_COST.json',{'new_units':sum(counts.values()),'counts':dict(counts),'GPU_seconds':time.monotonic()-start});(out/'EXIT_CODE').write_text('1\n');raise
 finally:done.set();ledger.close()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['val','replay'],required=True);main(a.parse_args().mode)
