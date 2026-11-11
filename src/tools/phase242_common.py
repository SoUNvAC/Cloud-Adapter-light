"""Phase241 stays read-only; every new artifact is confined to Phase242."""
from pathlib import Path
import collections,contextlib,hashlib,json,os,shutil,subprocess,time,sys
import numpy as np
R=Path('/home/scv/shared/phase242_20261031');P=Path('/home/scv/shared/phase241_20261030')
PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def protect_prior(event,args):
 protected=Path('/home/scv/shared/phase241_20261030')
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
  if path.is_relative_to(protected) and ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))):raise RuntimeError('Phase241 is read-only '+str(path))
 elif event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.replace'):
  for arg in args[:2 if event in ('os.rename','os.replace') else 1]:
   if isinstance(arg,(str,bytes,os.PathLike)) and Path(os.fsdecode(arg)).resolve().is_relative_to(protected):raise RuntimeError('Phase241 mutation denied')
sys.addaudithook(protect_prior)
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p=p if p.is_absolute() else R/p;assert p.is_relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False,default=str));tmp.replace(p)
def csvsave(p,rows):
 import csv
 keys=list(dict.fromkeys(k for r in rows for k in r));p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def cpu_seconds():return sum(js(p)['seconds'] for p in (R/'cpu_costs').glob('*.json'))
def hostfree():
 raw=subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True);v=json.loads(raw);v=[v] if isinstance(v,dict) else v;assert min(x['FreeSpace'] for x in v)>10*1024**3;return v
class CPU:
 def __init__(self,stage):
  self.stage=stage;self.start=time.monotonic();self.prior=cpu_seconds();self.path=R/'cpu_costs'/(stage+'.json');assert not self.path.exists();self.last_host=0;self.current=None;self.check()
 def check(self):
  import resource
  elapsed=self.prior+time.monotonic()-self.start;assert elapsed<43200,'12h cumulative CPU stage cap';rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert rss<16*1024**3,'16GiB CPU cap';size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());assert size<40*1024**3,'40GiB new work cap';free=shutil.disk_usage(R).free;assert free>10*1024**3,'10GiB reserve'
  if time.monotonic()-self.last_host>180:self.host_volumes=hostfree();self.last_host=time.monotonic()
  save('RUN_STATUS.json',{'PID':os.getpid(),'stage':self.stage,'current':self.current,'cumulative_CPU_seconds':elapsed,'peak_RSS_bytes':rss,'new_bytes':size,'free_bytes':free})
 def finish(self,status):
  import resource
  save(self.path,{'stage':self.stage,'status':status,'seconds':time.monotonic()-self.start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'host_volumes':getattr(self,'host_volumes',None)})
 def remaining(self):return max(1,43200-self.prior-(time.monotonic()-self.start))
def groups(split,domain,model):
 index=js(R/'INDEX_REFERENCES.json')[split];out=collections.defaultdict(list)
 for v in index:
  if v['domain']==domain and v['model']==model:out[v['scene']].append(v)
 return {s:sorted(v,key=lambda x:x['name']) for s,v in sorted(out.items())}
def scene_arrays(split,domain,scene,model,verify=True):
 rows=groups(split,domain,model)[scene];parts=[];truth=[]
 for v in rows:
  fp=P/v['score_file'];lp=P/v['label_file'];assert fp.is_relative_to(P) and lp.is_relative_to(P)
  if verify:assert sha(fp)==v['score_SHA'] and sha(lp)==v['label_SHA']
  p=np.load(fp,mmap_mode='r',allow_pickle=False);y=np.load(lp,mmap_mode='r',allow_pickle=False);assert p.dtype==np.float32 and p.shape[0]==3 and p.shape[1:]==y.shape;mask=y!=255;parts.append(np.asarray(p[:,mask]));truth.append(np.asarray(y[mask]));p._mmap.close();y._mmap.close()
 return np.concatenate(parts,axis=1),np.concatenate(truth),rows
def logscore(p):return np.log(np.maximum(np.asarray(p[2],np.float64),1e-12))-np.log(np.maximum(np.asarray(p[0],np.float64),1e-12))
def distribution(a):
 if not len(a):return {'n':0,'mean':None,'q10':None,'q25':None,'median':None,'q75':None,'q90':None}
 q=np.quantile(a,[.1,.25,.5,.75,.9]);return dict(n=len(a),mean=float(np.mean(a,dtype=np.float64)),**dict(zip(('q10','q25','median','q75','q90'),map(float,q))))
def cachepath(split,domain,model,scene,label):return R/'sorted'/split/domain/model/(scene+('_Shadow.f64' if label else '_Surface.f64'))
def sortedfile(path,values):
 path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();a=np.sort(np.asarray(values,np.float64))[::-1];a.astype('<f8',copy=False).tofile(path);return {'path':str(path.relative_to(R)),'n':len(a),'SHA':sha(path)}
def ranked(runs,cpu=None):
 folder=R/'scratch';folder.mkdir(exist_ok=True);manifest=folder/('rank_'+str(os.getpid())+'.manifest');assert not manifest.exists();manifest.write_text(''.join(f'{p} {y} {float(shift):.17g}\n' for p,y,shift in runs))
 try:return json.loads(subprocess.check_output([str(R/'code/rank242'),str(manifest)],text=True,timeout=cpu.remaining() if cpu else 60))
 finally:manifest.unlink()
def cm(y,pr):return np.bincount(3*y.astype(np.int64)+pr,minlength=9).reshape(3,3)
def ratio(a,b):return float(a/b) if b else None
def metrics(c):
 c=np.asarray(c,np.int64);ref=c.sum(1);pred=c.sum(0);d=c.diagonal();u=ref+pred-d;iou=[ratio(d[i],u[i]) for i in range(3)];o={'confusion':c.tolist(),'reference':ref.tolist(),'predicted':pred.tolist(),'IoU':iou,'mIoU':float(np.mean(iou)) if all(v is not None for v in iou) else None,'scene_mIoU_union':float(np.mean([iou[i] for i in range(3) if u[i]])) if u.sum() else None,'scene_mIoU_reference_present':float(np.mean([iou[i] for i in range(3) if ref[i]])) if ref.sum() else None,'Cloud_recall':ratio(d[1],ref[1]),'Shadow_recall':ratio(d[2],ref[2]),'Surface_to_Cloud_FPR':ratio(c[0,1],ref[0]),'nonShadow_to_Shadow_FPR':ratio(c[0,2]+c[1,2],ref[0]+ref[1])}
 for i in range(3):
  for j in range(3):
   if i!=j:o[f'{i}_to_{j}']=int(c[i,j]);o[f'{i}_to_{j}_rate']=ratio(c[i,j],ref[i])
 return o
