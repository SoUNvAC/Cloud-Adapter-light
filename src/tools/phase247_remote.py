"""Fixed L7 external diagnostic, no optimizer, fitting, or threshold selection."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,resource,shutil,argparse
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XFORMERS_DISABLED='1')
R=Path('/home/scv/shared/phase247_20261031')
D=Path('/home/scv/shared/phase246_20261031/integrated_finish_2015')
SRC=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def save(n,v):
 p=R/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(4194304):h.update(b)
 return h.hexdigest()
def infer():
 import numpy as np,rasterio
 from rasterio.windows import Window
 assert not (R/'INFERENCE_STARTED.json').exists(),'no blind inference restart'
 start=time.monotonic();data=js(D/'FINAL_DATA_READINESS.json');ident=js(R/'MODEL_IDENTITIES.json')
 assert len(data['results'])==3
 assert shutil.disk_usage(R).free>10*1024**3,'need free space before bounded 6GiB stage'
 assert sha(ident['Source_remote_path'])==ident['Source_SHA256']
 heads=[v for v in ident['C'] if v['domain']=='L8'];assert len(heads)==2
 for h in heads:assert sha(h['remote_path'])==h['head_SHA256']
 checks=[]
 for rec in data['results']:
  assert rec['same_native_grid'] and all(rec['identity_from_old_MTL'].values()) and rec['count_reconciliation']
  assert set(rec['unknown_codes'])<={191}
  for path,expected in rec['prepared_files'].items():assert sha(path)==expected['sha256']
  counts=np.zeros(256,np.int64);rgbpath=next(p for p in rec['prepared_files'] if 'RGB_native' in p);ypath=next(p for p in rec['prepared_files'] if 'reference_3class' in p)
  with rasterio.open(rec['native_paths']['mask']) as raw,rasterio.open(rgbpath) as rgb,rasterio.open(ypath) as labels:
   assert raw.transform==rgb.transform==labels.transform and raw.shape==rgb.shape==labels.shape and raw.crs==rgb.crs==labels.crs
   for _,win in labels.block_windows(1):
    y=labels.read(1,window=win);a=raw.read(1,window=win);x=rgb.read(window=win);expected=np.full(a.shape,255,np.uint8)
    expected[a==64]=2;expected[a==128]=0;expected[(a==255)|(a==192)]=1;expected[(x==0).any(0)]=255
    assert np.array_equal(y,expected);counts+=np.bincount(y.ravel(),minlength=256)
  assert [int(counts[i]) for i in range(3)]==[rec['mapped_class_counts'][k] for k in ['Surface','Cloud','Shadow']]
  checks.append({'scene':rec['scene'],'counts':{str(i):int(counts[i]) for i in [0,1,2,255]},'raw_191_ignored':rec['old2015_native_codes'].get('191',0),'same_native_grid':True,'independent_map_check':True})
 save('DATA_SUPPORT_ACCEPTANCE.json',{'scenes':checks,'qualified_support':'64 Shadow vs 128 Surface; 191 ignored, not full thin-cloud benchmark','full_2015_semantics_resolved':False})
 os.chdir(SRC);sys.path[:0]=[str(SRC/'tools'),str(SRC)]
 import torch
 from export_phase12_onnx import build_wrapper
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 calls=[0];depth=[0];ledger=(R/'CALL_LEDGER.jsonl').open('x',buffering=1)
 def pre(m,a):
  if depth[0]==0:
   assert calls[0]<2100 and time.monotonic()-start<1800
   calls[0]+=1;ledger.write(json.dumps({'ordinal':calls[0],'stage':context[0]})+'\n')
  depth[0]+=1
 def post(m,a,o):depth[0]-=1
 context=['construction'];save('INFERENCE_STARTED.json',{'epoch':time.time(),'PID':os.getpid()})
 hook1=torch.nn.modules.module.register_module_forward_pre_hook(pre);hook2=torch.nn.modules.module.register_module_forward_hook(post)
 wrapper,cfg=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=ident['Source_remote_path'],precision='fp32',active_block_indices=None))
 hook1.remove();hook2.remove();wrapper.eval().requires_grad_(False);model=wrapper.segmentor
 base={k:v.detach().cpu().clone() for k,v in model.decode_head.state_dict().items()}
 def digest_nonhead():
  h=hashlib.sha256()
  for k,v in sorted(model.state_dict().items()):
   if not k.startswith('decode_head.'):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 nonhead=digest_nonhead();states={'Source':base}
 for h in heads:states['C'+str(h['seed'])]=torch.load(h['remote_path'],map_location='cpu')['state_dict']
 tiles=js(D/'TILE_INDEX.json');index=[];samples=[];cms={};repeat=[]
 def forward(x,stage):
  assert calls[0]<2100 and time.monotonic()-start<1800
  calls[0]+=1;ledger.write(json.dumps({'ordinal':calls[0],'stage':stage})+'\n')
  with torch.inference_mode():p=wrapper(x).float().clamp_min(0);p=p/p.sum(1,keepdim=True).clamp_min(1e-12)
  a=p[0].cpu().numpy();assert a.shape==(3,512,512) and np.isfinite(a).all();return a
 for scene_i,rec in enumerate(data['results']):
  scene=rec['scene'];scene_tiles=[t for t in tiles if t['scene']==scene]
  rgbpath=next(p for p in rec['prepared_files'] if 'RGB_native' in p);ypath=next(p for p in rec['prepared_files'] if 'reference_3class' in p)
  for name,state in states.items():
   model.decode_head.load_state_dict(state,strict=True);cm=np.zeros((3,3),np.int64);prefix=f'{scene}_{name}';outfiles={k:R/(prefix+'_'+k+'.bin') for k in ['P','N']};streams={k:p.open('xb') for k,p in outfiles.items()};samples_for=[]
   with rasterio.open(rgbpath) as rgb,rasterio.open(ypath) as lbl:
    for ti,t in enumerate(scene_tiles):
     win=Window(t['x'],t['y'],t['width'],t['height']);raw=np.zeros((3,512,512),np.float32);raw[:,:t['height'],:t['width']]=rgb.read(window=win);raw=((raw-raw.min())/(raw.max()-raw.min()+1e-6)*255).astype(np.uint8)
     y=np.full((512,512),255,np.uint8);y[:t['height'],:t['width']]=lbl.read(1,window=win)
     x=torch.from_numpy(np.ascontiguousarray(raw[None]).astype(np.float32)).cuda();p=forward(x,scene+'/'+name+'/'+str(ti));score=(np.log(np.maximum(p[2],1e-12))-np.log(np.maximum(p[0],1e-12))).astype(np.float32)
     for k,c in [('P',2),('N',0)]:score[y==c].tofile(streams[k])
     valid=y!=255;pred=p.argmax(0);cm+=np.bincount((y[valid].astype(np.int64)*3+pred[valid]).ravel(),minlength=9).reshape(3,3)
     idx=np.arange(0,512*512,256);yy=y.ravel()[idx];keep=(yy==0)|(yy==2)
     samples_for.append(np.column_stack([np.full(int(keep.sum()),ti),idx[keep],yy[keep],p.reshape(3,-1)[:,idx[keep]].T,score.ravel()[idx[keep]]]))
     if scene_i==0 and ti==0:
      q=forward(x,'repeat/'+name);error=float(np.max(np.abs(q-p)));assert error<=1e-5;repeat.append({'model':name,'max_abs_difference':error})
     if ti%25==0:
      assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3
      assert torch.cuda.max_memory_reserved()<18*1024**3 and shutil.disk_usage(R).free>4*1024**3
      save('STATUS.json',{'stage':'inference','scene':scene,'model':name,'tile':ti,'tiles':len(scene_tiles),'calls':calls[0],'seconds':time.monotonic()-start})
     del x,p,score
   for f in streams.values():f.close()
   sample=R/(prefix+'_sample.npy');np.save(sample,np.concatenate(samples_for),allow_pickle=False);cms[scene+'/'+name]=cm.tolist()
   for k,path in outfiles.items():
    count=path.stat().st_size//4;assert count==rec['mapped_class_counts']['Shadow' if k=='P' else 'Surface'];index.append({'scene':scene,'model':name,'side':k,'path':str(path),'n':count,'sha256':sha(path)})
   assert digest_nonhead()==nonhead
   samples.append({'scene':scene,'model':name,'path':str(sample),'sha256':sha(sample)})
 save('SCORE_INDEX.json',index);save('SAMPLE_INDEX.json',samples);save('CONFUSIONS.json',cms)
 save('INFERENCE_COMPLETE.json',{'calls':calls[0],'seconds':time.monotonic()-start,'repeat':repeat,'nonhead_unchanged':True,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_peak_reserved':torch.cuda.max_memory_reserved(),'backend':'FP32; TF32 disabled','training':0,'fit':0,'thresholds':0});ledger.close()
def analyze():
 import numpy as np
 from sklearn.metrics import roc_auc_score,average_precision_score
 start=time.monotonic();index=js(R/'SCORE_INDEX.json');scenes=list(dict.fromkeys(v['scene'] for v in index));models=list(dict.fromkeys(v['model'] for v in index));matrices={};summary=[];sample_results=[];checks=[]
 for model in models:
  arrays={}
  for v in index:
   if v['model']!=model:continue
   assert sha(v['path'])==v['sha256'];a=np.fromfile(v['path'],dtype=np.float32);assert len(a)==v['n'] and np.isfinite(a).all();a.sort();arrays[v['scene'],v['side']]=a
  matrix=[]
  for i in scenes:
   p=arrays[i,'P']
   for j in scenes:
    n=arrays[j,'N'];wins=0.0
    for offset in range(0,len(p),1000000):
     q=p[offset:offset+1000000];lo=np.searchsorted(n,q,side='left');hi=np.searchsorted(n,q,side='right');wins+=float((lo.astype(np.float64)+(hi-lo)*.5).sum())
    auc=wins/(len(p)*len(n));matrix.append({'positive_scene':i,'negative_scene':j,'P':len(p),'N':len(n),'AUC':auc,'pairs':int(len(p))*int(len(n))})
  matrices[model]=matrix;within=[r for r in matrix if r['positive_scene']==r['negative_scene']];cross=[r for r in matrix if r['positive_scene']!=r['negative_scene']]
  def weighted(rows):return sum(r['AUC']*r['pairs'] for r in rows)/sum(r['pairs'] for r in rows)
  totalpairs=sum(r['pairs'] for r in matrix);assert totalpairs==sum(len(arrays[s,'P']) for s in scenes)*sum(len(arrays[s,'N']) for s in scenes)
  summary.append({'model':model,'scene_mean_AUC':float(np.mean([r['AUC'] for r in within])),'within_weighted_AUC':weighted(within),'cross_AUC':weighted(cross),'pooled_AUC':weighted(matrix),'total_pairs':totalpairs})
  del arrays
 samples=js(R/'SAMPLE_INDEX.json');lookup={(v['scene'],v['model']):v for v in samples}
 for scene in scenes:
  base=np.load(lookup[scene,'Source']['path']);y=base[:,2]==2
  for model in models:
   a=np.load(lookup[scene,model]['path']);assert np.array_equal(base[:,:3],a[:,:3]);assert sha(lookup[scene,model]['path'])==lookup[scene,model]['sha256']
   delta=a[:,6]-base[:,6];record={'scene':scene,'model':model,'sample_n':len(a),'sample_p':int(y.sum()),'sample_pair_AUC':float(roc_auc_score(y,a[:,6])),'sample_pshadow_AUC':float(roc_auc_score(y,a[:,5])),'sample_pshadow_AP':float(average_precision_score(y,a[:,5]))}
   for c,k in [(0,'Surface'),(2,'Shadow')]:record[k+'_delta_q25_q50_q75']=np.quantile(delta[a[:,2]==c],[.25,.5,.75]).tolist()
   sample_results.append(record)
   neg=base[~y,6];pos=base[y,6];alt=float((np.searchsorted(np.sort(neg),pos,'left')+np.searchsorted(np.sort(neg),pos,'right')).sum()*.5/(len(pos)*len(neg)));assert abs(alt-roc_auc_score(y,base[:,6]))<1e-12;checks.append({'scene':scene,'model':model,'sample_AUC_independent_match':True})
 # Explicit synthetic ties and unequal scene supports protect the central pair decomposition.
 p=np.array([0,1,1,3],np.float32);n=np.array([0,1,2],np.float32);hand=np.mean([float(x>z)+.5*float(x==z) for x in p for z in n]);auc=roc_auc_score([1]*len(p)+[0]*len(n),np.concatenate([p,n]));assert abs(hand-auc)<1e-12
 save('RESULT_TABLES.json',{'summary':summary,'matrices':matrices,'same_pixel_sample':sample_results,'metric_unit':'fraction; multiply 100 for percentage','analysis_seconds':time.monotonic()-start})
 save('VALIDATION.json',{'passed':True,'sample_AUC_checks':checks,'synthetic_ties':True,'full_score_hashes':len(index),'same_support':True,'scene_pairs_not_independent':True})
 save('STATUS.json',{'stage':'complete','new_training':0,'new_fits':0})
if __name__=='__main__':
 try:
  infer();analyze()
 except BaseException:
  save('FAILURE.json',{'traceback':traceback.format_exc()});save('STATUS.json',{'stage':'failed'});raise
