"""One frozen source-val pass: same features, teacher and eight alpha.25 heads."""
from pathlib import Path
import argparse,collections,copy,hashlib,json,os,resource,subprocess,sys,threading,time,traceback
R=Path('/home/scv/shared/phase223_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False,default=str));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>=4*1024**3;return v
def main():
 assert not (R/'EXECUTION_STARTED.json').exists();os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1';lock=js(R/'INPUT_MODEL_LOCK.json');assert lock['expected_units']==5360
 for n,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/n)==h
 for n,h in lock['runtime_SHA'].items():assert sha(SRC/n)==h,n
 assert sha(lock['source_path'])==lock['Source_SHA']
 for v in lock['models'].values():assert sha(v['endpoint_path'])==v['endpoint_SHA'] and sha(v['interpolant_path'])==v['interpolant_SHA']
 for n,h in lock['expected_SHA'].items():assert sha(R/'expected'/n)==h
 for v in lock['support']:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
 import numpy as np,torch
 import torch.nn.functional as F
 from export_phase12_onnx import build_wrapper,StaticMask2FormerLogits
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SOURCE_TO_PARENT
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
 counts=collections.Counter();stage='build';current=None;start=time.monotonic();done=threading.Event();samples=[];denied=[];access=collections.Counter();ledger=(R/'CALL_LEDGER.jsonl').open('x',buffering=1)
 save('PID',os.getpid());save('EXECUTION_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'expected_units':5360,'cap_units':6000,'GPU_seconds_cap':3600,'host_volumes':host(),'optimizer_updates':0})
 def status():save('RUN_STATUS.json',{'stage':stage,'current':current,'counts':dict(counts),'units':sum(counts.values()),'GPU_seconds':time.monotonic()-start,'PID':os.getpid()})
 def count(k,key=None):
  assert sum(counts.values())<6000 and time.monotonic()-start<3600;counts[k]+=1;ledger.write(json.dumps({'ordinal':sum(counts.values()),'kind':k,'model':key,'input':current,'stage':stage})+'\n')
 def guard():
  last=0
  while not done.wait(15):
   try:
    v={'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,'GPU_reserved_peak':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'persistent_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))};assert v['CPU_peak_bytes']<16*1024**3 and max(v['GPU_allocated_peak'],v['GPU_reserved_peak'])<20*1024**3 and v['persistent_bytes']<2000000000 and time.monotonic()-start<3600
    if time.monotonic()-last>120:v['host_volumes']=host();last=time.monotonic()
    samples.append(v);save('RESOURCE_SAMPLES.json',samples);status()
   except BaseException:(R/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(R/'EXIT_CODE').write_text('71\n');os._exit(71)
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=os.path.abspath(os.fsdecode(args[0])).replace('\\','/')
  if any(x in p for x in ['cloudsen12_high_l1c/ann_dir/train/','cloudsen12_high_l1c/img_dir/train/','cloudsen12_high_l1c/ann_dir/test/','cloudsen12_high_l1c/img_dir/test/','/data/l8_biome/','/data/sparcs/']):denied.append(p);raise RuntimeError('223 forbidden pixel input '+p)
  if 'cloudsen12_high_l1c/ann_dir/val/' in p:access['source_val_reference_opens']+=1
 sys.addaudithook(audit);threading.Thread(target=guard,daemon=True).start()
 try:
  wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=lock['source_path'],precision='fp32',active_block_indices=None));model=wrapper.segmentor.eval()
  for p in model.parameters():p.requires_grad=False
  ck=torch.load(lock['source_path'],map_location='cpu')['state_dict'];allsource=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,allsource[n].detach().cpu()) for n,t in ck.items())
  initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};heads={'Source':model.decode_head};receipts={}
  for key,v in lock['models'].items():
   end=torch.load(v['endpoint_path'],map_location='cpu');assert end['seed']==v['seed'] and end['arm']==v['arm'] and end['steps']==4000;state={n:(t.double()+.25*(end['state_dict'][n].double()-t.double())).to(t.dtype) for n,t in initial.items()};old=torch.load(v['interpolant_path'],map_location='cpu')['state_dict'];assert set(old)==set(state) and all(torch.equal(t,old[n]) for n,t in state.items());h=copy.deepcopy(model.decode_head).eval();h.load_state_dict(state)
   for p in h.parameters():p.requires_grad=False
   heads[key]=h;receipts[key]={'endpoint_SHA':v['endpoint_SHA'],'interpolant_SHA':v['interpolant_SHA'],'all_alpha025_tensors_match':True}
  clone=copy.deepcopy(heads['Source']).eval();heads['Source_clone']=clone;official={}
  for key in ['Source']+list(lock['models']):
   proxy=copy.copy(model);proxy._modules=dict(model._modules);proxy.decode_head=heads[key];official[key]=StaticMask2FormerLogits(proxy,'fp32').cuda().eval()
  def tensors(h):return dict(list(h.named_parameters())+list(h.named_buffers()))
  def digest(d):
   h=hashlib.sha256()
   for n,t in sorted(d.items()):h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
   return h.hexdigest()
  frozen={'full_Source':{n:t.detach().cpu().clone() for n,t in tensors(model).items()},**{k:{n:t.detach().cpu().clone() for n,t in tensors(h).items()} for k,h in heads.items()}};digests={k:digest(v) for k,v in frozen.items()};save('FROZEN_INITIAL.json',{'digests':digests,'heads':receipts,'all_eval':True,'optimizer_updates':0})
  def frozen_check():
   for key,values in frozen.items():
    h=model if key=='full_Source' else heads[key];now=tensors(h);assert all(torch.equal(t,now[n].detach().cpu()) for n,t in values.items()) and not h.training and all(not p.requires_grad and p.grad is None for p in h.parameters())
  def semantic(h,features,kind,key):
   count(kind,key);c,m=h(features,None);return torch.einsum('bqc,bqhw->bchw',F.softmax(c[-1].float(),-1)[...,:-1],F.interpolate(m[-1].float(),size=(512,512),mode='bilinear',align_corners=False).sigmoid())
  def pred(s):
   z=s.float().clamp_min(0);return (z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
  def rawsha(x):return hashlib.sha256(x.tobytes()).hexdigest()
  expected={k:js(R/'expected'/f'{k}.json') for k in ['Source']+list(lock['models'])};byname={k:{v['name']:v for v in e['per_input']} for k,e in expected.items()};predhash={k:hashlib.sha256() for k in expected};cm=collections.defaultdict(lambda:np.zeros((3,3),np.int64));gap_arrays=collections.defaultdict(list);gap_samples=[];seen_gap=set();coordinate=[];negmin=0.;stage='formal_sourceval'
  with (R/'PER_IMAGE.jsonl').open('x',buffering=1) as per,(R/'JOINT_COUNTS.jsonl').open('x',buffering=1) as joints:
   for index,row in enumerate(lock['support'],1):
    current={'index':index,'name':row['name'],'ROI':row['scene']};x=torch.from_numpy(load_rgb_image(Path(row['image_path']),512)).cuda();count('shared_backbone');features=model.extract_feat(wrapper.normalize(x));feature_digest=digest({str(i):t for i,t in enumerate(features)}) if index==1 else None;sem={'Source':semantic(heads['Source'],features,'teacher_head','Source')};sem.update({k:semantic(heads[k],features,'student_head',k) for k in lock['models']})
    lp=F.log_softmax(sem['Source'].float(),1);prob=lp.exp();tp=pred(sem['Source']);y=load_parent_target(Path(row['mask_path']),SOURCE_TO_PARENT,(512,512));valid=y!=255;tc=tp==y;confidence=prob.max(1).values[0].cpu().numpy();preds={'Source':tp}
    if index==1:
     zs=semantic(clone,features,'Source_clone_head','Source_clone');lv=(prob*(lp-F.log_softmax(zs.float(),1))).sum(1);assert abs(float(lv.max()))<=1e-6 and abs(float(lv.min()))<=1e-6;sem['Source_clone']=zs
     for key,w in official.items():count('official_full_segmentor',key);assert torch.equal(w(x),sem[key])
     assert feature_digest==digest({str(i):t for i,t in enumerate(features)});save('ENGINEERING_GATE_ACCEPTED.json',{'name':row['name'],'Source_clone_KL_min':float(lv.min()),'Source_clone_KL_max':float(lv.max()),'nine_official_wrappers_semantic_bitwise_equal':True,'feature_unchanged_by_heads':True,'formal_first_input_reused':True,'extra_units':10,'updates':0})
    for key,v in lock['models'].items():
     lq=F.log_softmax(sem[key].float(),1);vv=(prob*(lp-lq)).sum(1)[0];assert torch.isfinite(vv).all() and float(vv.min())>=-1e-6;arr=vv.cpu().numpy();sp=pred(sem[key]);preds[key]=sp;dis=sp!=tp;sc=sp==y
     def stats(mask):
      n=int(mask.sum());ks=float(arr[mask].sum(dtype=np.float64));return {'pixels':n,'KL_sum':ks,'KL_mean':ks/n if n else None,'teacher_disagreements':int((dis&mask).sum()),'student_errors':int(((sp!=y)&mask).sum()),'teacher_errors':int(((tp!=y)&mask).sum())}
     result={'model':key,'target':v['target'],'seed':v['seed'],'arm':v['arm'],'alpha':.25,'pool':'Source_val535','name':row['name'],'ROI':row['scene'],'geometry_pixels':arr.size,'KL_sum_geometry':float(arr.sum(dtype=np.float64)),'KL_mean_geometry':float(arr.mean(dtype=np.float64)),'KL_min':float(arr.min()),'KL_max':float(arr.max()),'negative_KL_pixels':int((arr<0).sum()),'negative_KL_sum':float(arr[arr<0].sum(dtype=np.float64)),'teacher_disagreements_geometry':int(dis.sum()),'teacher_student_geometry_counts':np.bincount(3*tp.ravel()+sp.ravel(),minlength=9).reshape(3,3).tolist(),'teacher_prediction_SHA':rawsha(tp),'student_prediction_SHA':rawsha(sp),'reference_valid':stats(valid),'reference_ignore_pixels':int((~valid).sum()),'reference_classes':[{'class':c,**stats(valid&(y==c))} for c in range(3)],'correctness':[{'teacher_correct':t,'student_correct':s,**stats(valid&(tc==t)&(sc==s))} for t in [True,False] for s in [True,False]],'both_wrong_subtypes':[{'same_error_class':same,**stats(valid&~tc&~sc&((tp==sp)==same))} for same in [True,False]],'reference_teacher_student_counts':np.bincount(9*y[valid].astype(np.int64)+3*tp[valid]+sp[valid],minlength=27).reshape(3,3,3).tolist(),'error_flows':[{'reference':c,'student':d,**stats(valid&(y==c)&(sp==d))} for c in range(3) for d in range(3) if c!=d]}
     shadow=[]
     for dest in [0,1]:
      keep=valid&(y==2)&(tp==2)&(sp==dest);n=int(keep.sum());q=stats(keep);q.update(destination=dest,teacher_confidence_sum=float(confidence[keep].sum(dtype=np.float64)),teacher_confidence_mean=float(confidence[keep].mean(dtype=np.float64)) if n else None);shadow.append(q)
      if dest==0 and n:
       tg=(sem['Source'][0,2]-sem['Source'][0,0]).cpu().numpy()[keep].copy();sg=(sem[key][0,2]-sem[key][0,0]).cpu().numpy()[keep].copy();gap_arrays[key,row['scene'],'teacher'].append(tg);gap_arrays[key,row['scene'],'student'].append(sg)
       if (key,row['scene']) not in seen_gap:
        yy,xx=np.argwhere(keep)[0];gap_samples.append({'model':key,'name':row['name'],'ROI':row['scene'],'yx':[int(yy),int(xx)],'reference':2,'teacher_prediction':2,'student_prediction':0,'teacher_S':sem['Source'][0,:,yy,xx].tolist(),'student_S':sem[key][0,:,yy,xx].tolist(),'teacher_gap':float(tg[0]),'student_gap':float(sg[0])});seen_gap.add((key,row['scene']))
     result['teacher_correct_shadow_destinations']=shadow;per.write(json.dumps(result,allow_nan=False)+'\n');negmin=min(negmin,result['KL_min'])
     if index==1:
      for yy,xx in lock['fixed_coordinates']:coordinate.append({'model':key,'yx':[yy,xx],'teacher_S':sem['Source'][0,:,yy,xx].tolist(),'student_S':sem[key][0,:,yy,xx].tolist(),'p':prob[0,:,yy,xx].tolist(),'logp':lp[0,:,yy,xx].tolist(),'logq':lq[0,:,yy,xx].tolist(),'KL_GPU':float(vv[yy,xx])})
    for key,prediction in preds.items():
     c=np.bincount(3*y[valid].astype(np.int64)+prediction[valid],minlength=9).reshape(3,3);old=byname[key][row['name']];assert c.tolist()==old['confusion'],('old exact confusion mismatch',key,row['name']);cm[key,row['scene']]+=c;predhash[key].update(prediction.tobytes())
     if 'prediction_SHA' in old:assert rawsha(prediction)==old['prediction_SHA']
    for target in ['SPARCS','L8']:
     for s in [21701,21702]:
      c=preds[f'{target}_{s}_C'];d=preds[f'{target}_{s}_D'];j=np.bincount(27*y[valid].astype(np.int64)+9*tp[valid]+3*c[valid]+d[valid],minlength=81).reshape(3,3,3,3);joints.write(json.dumps({'target':target,'seed':s,'name':row['name'],'ROI':row['scene'],'valid_pixels':int(valid.sum()),'J':j.tolist(),'teacher_prediction_SHA':rawsha(tp),'C_prediction_SHA':rawsha(c),'D_prediction_SHA':rawsha(d)})+'\n')
    if index%25==0:frozen_check();status()
    del features,sem,lp,prob,preds
  reproduction={}
  for key,e in expected.items():
   assert {g:cm[key,g].tolist() for g in e['per_group']}==e['per_group'];expected_pred=e.get('prediction_SHA',e.get('prediction_sha256'));assert expected_pred and predhash[key].hexdigest()==expected_pred,(key,'prediction digest');reproduction[key]={'all535_input_confusions_exact':True,'all107_ROI_confusions_exact':True,'pooled_confusion_exact':sum(cm[key,g] for g in e['per_group']).tolist()==e['confusion'],'prediction_SHA':predhash[key].hexdigest(),'expected_SHA':expected_pred}
  with (R/'SIGNED_GAP_SUMMARY.jsonl').open('x') as out:
   for key in lock['models']:
    rois=sorted({v['scene'] for v in lock['support']})
    for roi in rois+['__pooled__']:
     stat={}
     for kind in ['teacher','student']:
      pieces=[a for g in rois for a in gap_arrays.get((key,g,kind),[])] if roi=='__pooled__' else gap_arrays.get((key,roi,kind),[]);v=np.concatenate(pieces).astype(np.float64) if pieces else np.empty(0);stat[kind]={'pixels':len(v),'P10':float(np.quantile(v,.1)) if len(v) else None,'median':float(np.quantile(v,.5)) if len(v) else None,'P90':float(np.quantile(v,.9)) if len(v) else None}
     assert stat['teacher']['pixels']==stat['student']['pixels'];out.write(json.dumps({'model':key,'ROI':roi,'condition':'reference Shadow AND teacher Shadow AND student Surface','gap':'S_Shadow-S_Surface','quantile':'linear exact all selected pixels','stats':stat})+'\n')
  frozen_check();assert counts=={'shared_backbone':535,'teacher_head':535,'student_head':4280,'Source_clone_head':1,'official_full_segmentor':9};assert sha(lock['source_path'])==lock['Source_SHA'];done.set();stage='complete';current=None
  save('REPRODUCTION_ACCEPTED.json',reproduction);save('FIXED_KL_COORDINATES.json',coordinate);save('SIGNED_GAP_FIXED_SAMPLES.json',gap_samples);save('HEAD_INTERPOLATION_ACCEPTED.json',receipts);save('FROZEN_FINAL.json',{'all_tensors_exact':True,'initial_digests':digests,'final_digests':{'full_Source':digest(tensors(model)),**{k:digest(tensors(h)) for k,h in heads.items()}},'optimizer_updates':0,'all_no_grad_eval':True});save('ACCESS_AUDIT.json',{'forbidden_attempts':denied,'source_val_reference_opens':access['source_val_reference_opens'],'source_train_replay_pixel_reads':0,'source_train_test_reference_reads':0,'source_official_test_reads':0,'target_pixel_reads':0});save('COMPLETE.json',{'status':'complete','total_units':sum(counts.values()),'counts':dict(counts),'GPU_seconds':time.monotonic()-start,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),'host_volumes_final':host(),'new_fits':0,'per_image_records':4280,'joint_records':2140,'negative_KL_min':negmin});status();(R/'EXIT_CODE').write_text('0\n')
 except BaseException:
  done.set();(R/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(counts),'total_units':sum(counts.values()),'GPU_seconds':time.monotonic()-start,'updates':0});status();(R/'EXIT_CODE').write_text('1\n');raise
 finally:done.set();ledger.close()
if __name__=='__main__':main()
