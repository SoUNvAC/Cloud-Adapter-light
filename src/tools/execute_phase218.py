"""Fixed final-head KL diagnosis only. No fitting, gradients, target evaluation or downloads."""
from pathlib import Path
import argparse,collections,copy,hashlib,json,os,resource,subprocess,sys,threading,time,traceback
R=Path('/home/scv/shared/phase218_20261023');SRC=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=str));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>=4*1024**3;return v
def main():
 assert not (R/'EXECUTION_STARTED.json').exists(),'One run only; retain failures/costs before any recovery'
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 lock=js(R/'INPUT_MODEL_LOCK.json');assert lock['expected_units']==4758
 assert sha(lock['source_path'])==lock['Source_SHA']
 for n,h in lock['runtime_SHA'].items():assert sha(SRC/n)==h,n
 for n,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/n)==h,n
 for n,h in lock['expected_SHA'].items():assert sha(R/'expected217'/n)==h
 for m in lock['models'].values():assert sha(m['path'])==m['SHA']
 for pool,rows in lock['pools'].items():
  for row in rows:
   assert sha(row['image_path'])==row['image_SHA']
   if pool=='Source_val535':assert sha(row['mask_path'])==row['mask_SHA']
 import numpy as np,torch
 import torch.nn.functional as F
 from export_phase12_onnx import build_wrapper,StaticMask2FormerLogits
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SOURCE_TO_PARENT
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False;torch.set_grad_enabled(False)
 counts=collections.Counter();stage='build';current=None;start=time.monotonic();done=threading.Event();samples=[];access=collections.Counter();denied=[]
 ledger=(R/'CALL_LEDGER.jsonl').open('x',buffering=1)
 save('PID',os.getpid());save('EXECUTION_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'code_SHA':sha(Path(__file__)),'expected_units':4758,'calls_cap':6000,'GPU_seconds_cap':3600,'host_volumes':host(),'optimizer_updates':0})
 def status():save('RUN_STATUS.json',{'stage':stage,'current':current,'counts':dict(counts),'units':sum(counts.values()),'GPU_seconds':time.monotonic()-start,'PID':os.getpid()})
 def count(kind,model=None):
  assert sum(counts.values())<6000 and time.monotonic()-start<3600
  counts[kind]+=1;ordinal=sum(counts.values());ledger.write(json.dumps({'ordinal':ordinal,'kind':kind,'model':model,'stage':stage,'input':current})+'\n');return ordinal
 def guard():
  last=0
  while not done.wait(15):
   try:
    sample={'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,'GPU_reserved_peak':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'persistent_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.name.endswith('.tmp')),'epoch':time.time()}
    assert sample['CPU_peak_bytes']<16*1024**3 and max(sample['GPU_allocated_peak'],sample['GPU_reserved_peak'])<20*1024**3 and sample['persistent_bytes']<2000000000 and time.monotonic()-start<3600
    if time.monotonic()-last>120:sample['host_volumes']=host();last=time.monotonic()
    samples.append(sample);save('RESOURCE_SAMPLES.json',samples);status()
   except BaseException:
    (R/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(R/'EXIT_CODE').write_text('71\n');os._exit(71)
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=os.path.abspath(os.fsdecode(args[0])).replace('\\','/')
  if any(s in p for s in ['cloudsen12_high_l1c/ann_dir/train/','cloudsen12_high_l1c/ann_dir/test/','cloudsen12_high_l1c/img_dir/test/','/data/sparcs/','/data/L8/']):denied.append(p);raise RuntimeError('218 forbidden pixel access: '+p)
  if 'cloudsen12_high_l1c/ann_dir/val/' in p:access['val_reference_opens']+=1
 sys.addaudithook(audit);threading.Thread(target=guard,daemon=True).start()
 try:
  wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=lock['source_path'],precision='fp32',active_block_indices=None));model=wrapper.segmentor.eval()
  for p in model.parameters():p.requires_grad=False
  ck=torch.load(lock['source_path'],map_location='cpu')['state_dict'];allsource=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,allsource[n].detach().cpu()) for n,t in ck.items())
  heads={'Source':model.decode_head}
  for key,m in lock['models'].items():
   h=copy.deepcopy(model.decode_head).eval();state=torch.load(m['path'],map_location='cpu');assert state['steps']==4000 and state['seed']==m['seed'] and state['arm']==m['arm'];h.load_state_dict(state['state_dict'],strict=True)
   for p in h.parameters():p.requires_grad=False
   assert all(torch.equal(t,h.state_dict()[n].detach().cpu()) for n,t in state['state_dict'].items());heads[key]=h
  clone=copy.deepcopy(heads['Source']).eval();heads['Source_clone']=clone
  official={}
  for key in ['Source']+list(lock['models']):
   proxy=copy.copy(model);proxy._modules=dict(model._modules);proxy.decode_head=heads[key]
   official[key]=StaticMask2FormerLogits(proxy,'fp32').cuda().eval()
  def tensors(module):return dict(list(module.named_parameters())+list(module.named_buffers()))
  initial={'full_Source':{n:t.detach().cpu().clone() for n,t in tensors(model).items()},**{k:{n:t.detach().cpu().clone() for n,t in tensors(h).items()} for k,h in heads.items()}}
  def digest(d):
   h=hashlib.sha256()
   for n,t in sorted(d.items()):h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
   return h.hexdigest()
  initial_sha={k:digest(v) for k,v in initial.items()};save('FROZEN_INITIAL.json',{'digests':initial_sha,'all_eval':True,'trainable_parameters':0,'Source_checkpoint_tensors_exact':len(ck),'loaded_heads':lock['models'],'teacher_gradient':False})
  def frozen():
   for k,values in initial.items():
    mod=model if k=='full_Source' else heads[k];now=tensors(mod);assert set(now)==set(values) and all(torch.equal(t,now[n].detach().cpu()) for n,t in values.items()),k
    assert not mod.training and all(not p.requires_grad and p.grad is None for p in mod.parameters())
  def semantic(head,features,kind,key):
   count(kind,key);c,m=head(features,None);return torch.einsum('bqc,bqhw->bchw',F.softmax(c[-1].float(),-1)[...,:-1],F.interpolate(m[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid())
  def predict(s):
   z=s.float().clamp_min(0);return (z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
  def outputs(row,engineering=False):
   x=torch.from_numpy(load_rgb_image(Path(row['image_path']),512)).cuda();ordinal=count('shared_backbone');features=model.extract_feat(wrapper.normalize(x));before=digest({str(i):f for i,f in enumerate(features)}) if engineering else None
   sem={'Source':semantic(heads['Source'],features,'teacher_head','Source')}
   sem.update({k:semantic(heads[k],features,'student_head',k) for k in lock['models']})
   if engineering:
    sem['Source_clone']=semantic(clone,features,'Source_clone_head','Source_clone')
    assert before==digest({str(i):f for i,f in enumerate(features)}),'head modified shared features'
    for k,w in official.items():
     count('official_full_segmentor',k);s=w(x);assert torch.equal(s,sem[k]),('shared versus official semantic drift',k);assert np.array_equal(predict(s),predict(sem[k]))
   return {'semantic':sem,'feature_SHA':before,'feature_backbone_ordinal':ordinal}
  def kl(s,t):
   lp=F.log_softmax(t.float(),1);lq=F.log_softmax(s.float(),1);p=lp.exp();v=(p*(lp-lq)).sum(1)[0];assert torch.isfinite(v).all() and float(v.min())>=-lock['negative_KL_tolerance'];return v,p,lp,lq
  cached={};gate=[];formula=[];stage='engineering_gate'
  with torch.inference_mode():
   for pool,rows in lock['pools'].items():
    row=rows[0];current={'pool':pool,'name':row['name']};data=outputs(row,True);cached[pool,row['name']]=data;sem=data['semantic'];v,_,_,_=kl(sem['Source_clone'],sem['Source']);assert abs(float(v.max()))<1e-6 and abs(float(v.min()))<1e-6
    gate.append({'pool':pool,'name':row['name'],'image_SHA':row['image_SHA'],'feature_SHA':data['feature_SHA'],'feature_backbone_ordinal':data['feature_backbone_ordinal'],'Source_clone_KL_min':float(v.min()),'Source_clone_KL_max':float(v.max()),'five_official_wrappers_semantic_and_argmax_bitwise_equal':True})
    for key in list(lock['models'])+['Source_clone']:
     v,p,lp,lq=kl(sem[key],sem['Source'])
     for y,x in lock['fixed_coordinates']:formula.append({'pool':pool,'name':row['name'],'model':key,'yx':[y,x],'Source_S':sem['Source'][0,:,y,x].tolist(),'student_S':sem[key][0,:,y,x].tolist(),'p':p[0,:,y,x].tolist(),'q':lq.exp()[0,:,y,x].tolist(),'logp':lp[0,:,y,x].tolist(),'logq':lq[0,:,y,x].tolist(),'KL_GPU':float(v[y,x])})
   assert gate[0]['image_SHA']!=gate[1]['image_SHA'] and gate[0]['feature_SHA']!=gate[1]['feature_SHA'] and gate[0]['feature_backbone_ordinal']!=gate[1]['feature_backbone_ordinal'];frozen()
   save('ENGINEERING_GATE_ACCEPTED.json',{'checks':gate,'fresh_features_for_two_different_inputs':True,'formal_first_inputs_reused':True,'no_reference_labels_in_gate':True,'extra_units':12,'units_now':sum(counts.values())});save('FIXED_KL_COORDINATES.json',formula)
   expected={'Source':js(R/'expected217/Source__source.json'),**{k:js(R/f'expected217/{k}_a1__source.json') for k in lock['models']}}
   by_input={k:{r['name']:r for r in v['per_input']} for k,v in expected.items()};predhash={k:hashlib.sha256() for k in expected};cm_total={k:np.zeros((3,3),np.int64) for k in expected};cm_rois={k:collections.defaultdict(lambda:np.zeros((3,3),np.int64)) for k in expected};formal_images=collections.Counter();stage='formal_scoring'
   with (R/'PER_IMAGE.jsonl').open('x',buffering=1) as stream:
    for pool,rows in lock['pools'].items():
     for index,row in enumerate(rows,1):
      current={'pool':pool,'index':index,'name':row['name']};data=cached.pop((pool,row['name']),None)
      if data is None:data=outputs(row)
      sem=data['semantic'];teacher=predict(sem['Source']);reference=None;valid=None
      if pool=='Source_val535':
       reference=load_parent_target(Path(row['mask_path']),SOURCE_TO_PARENT,(512,512));valid=reference!=255
       for k in expected:
        pred=teacher if k=='Source' else predict(sem[k]);c=np.bincount(3*reference[valid]+pred[valid],minlength=9).reshape(3,3);assert c.tolist()==by_input[k][row['name']]['confusion'],('217 per-input mismatch',k,row['name']);cm_total[k]+=c;cm_rois[k][row['ROI']]+=c;predhash[k].update(pred.tobytes())
      for key,meta in lock['models'].items():
       vv,_,_,_=kl(sem[key],sem['Source']);arr=vv.cpu().numpy();student=predict(sem[key]);dis=student!=teacher;total=arr.size
       def stats(mask):
        n=int(mask.sum());return {'pixels':n,'KL_sum':float(arr[mask].sum(dtype=np.float64)),'KL_mean':float(arr[mask].mean(dtype=np.float64)) if n else None,'teacher_disagreements':int((dis&mask).sum()),'student_errors':int(((student!=reference)&mask).sum()) if reference is not None else None,'teacher_errors':int(((teacher!=reference)&mask).sum()) if reference is not None else None}
       result={'pool':pool,'name':row['name'],'ROI':row['ROI'],'seed':meta['seed'],'arm':meta['arm'],'geometry_pixels':total,'KL_sum_geometry':float(arr.sum(dtype=np.float64)),'KL_mean_geometry':float(arr.mean(dtype=np.float64)),'negative_KL_pixels':int((arr<0).sum()),'KL_min':float(arr.min()),'KL_max':float(arr.max()),'negative_KL_sum':float(arr[arr<0].sum(dtype=np.float64)),'teacher_disagreements_geometry':int(dis.sum()),'teacher_prediction_SHA':hashlib.sha256(teacher.tobytes()).hexdigest(),'student_prediction_SHA':hashlib.sha256(student.tobytes()).hexdigest(),'teacher_student_geometry_counts':np.bincount(3*teacher.ravel()+student.ravel(),minlength=9).reshape(3,3).tolist()}
       if reference is not None:
        tc=teacher==reference;sc=student==reference;result['reference_valid']=stats(valid);result['reference_ignore_pixels']=int((~valid).sum());result['reference_classes']=[{'class':k,**stats(valid&(reference==k))} for k in range(3)];result['correctness']=[{'teacher_correct':t,'student_correct':s,**stats(valid&(tc==t)&(sc==s))} for t in [True,False] for s in [True,False]];result['both_wrong_subtypes']=[{'same_error_class':same,**stats(valid&~tc&~sc&((teacher==student)==same))} for same in [True,False]];result['reference_teacher_student_counts']=np.bincount(9*reference[valid]+3*teacher[valid]+student[valid],minlength=27).reshape(3,3,3).tolist()
       stream.write(json.dumps(result,allow_nan=False)+'\n')
      formal_images[pool]+=1
      if index%50==0:frozen();status()
   reproduction={}
   for key,e in expected.items():
    assert cm_total[key].tolist()==e['confusion'] and {r:c.tolist() for r,c in cm_rois[key].items()}==e['per_group'] and predhash[key].hexdigest()==e['prediction_SHA'],('217 final reproduction',key)
    reproduction[key]={'all535_input_confusions_exact':True,'all107_ROI_confusions_exact':True,'pooled_confusion_exact':True,'prediction_SHA':predhash[key].hexdigest(),'expected_SHA':e['prediction_SHA']}
   assert formal_images=={'Replay256':256,'Source_val535':535} and not cached
   assert counts=={'shared_backbone':791,'teacher_head':791,'student_head':3164,'Source_clone_head':2,'official_full_segmentor':10}
   frozen();assert sha(lock['source_path'])==lock['Source_SHA'] and all(sha(m['path'])==m['SHA'] for m in lock['models'].values());done.set();stage='complete';current=None
   save('REPRODUCTION_ACCEPTED.json',reproduction);save('FROZEN_FINAL.json',{'initial_digests':initial_sha,'final_digests':{'full_Source':digest(tensors(model)),**{k:digest(tensors(h)) for k,h in heads.items()}},'all_tensors_exact':True,'all_no_grad_and_eval':True,'optimizer_updates':0,'parameter_requires_grad_count':0})
   save('ACCESS_AUDIT.json',{'forbidden_attempts':denied,'val_reference_opens':access['val_reference_opens'],'source_train_reference_reads':0,'source_official_test_reads':0,'target_evaluation_reads':0})
   save('COMPLETE.json',{'status':'complete','counts':dict(counts),'total_units':sum(counts.values()),'GPU_seconds':time.monotonic()-start,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'formal_images':dict(formal_images),'formal_rows':3164,'host_volumes_final':host(),'initial_gate_inputs_reused':True,'precision':'FP32 semantic/softmax/KL; float64 statistics accumulation','evaluation_flags':{'deterministic_algorithms':False,'cudnn_deterministic':False,'cudnn_benchmark':False},'persistent_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())});status();(R/'EXIT_CODE').write_text('0\n')
 except BaseException:
  done.set();(R/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(counts),'units':sum(counts.values()),'GPU_seconds':time.monotonic()-start});status();(R/'EXIT_CODE').write_text('1\n');raise
 finally:done.set();ledger.close()
if __name__=='__main__':main()
