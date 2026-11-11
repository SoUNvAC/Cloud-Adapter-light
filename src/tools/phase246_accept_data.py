"""No-model, blockwise native RGB/reference support inspection for fixed L7 intake."""
from pathlib import Path
import json, re, tarfile, shutil, hashlib, math, os, resource, time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

ROOT=Path('/home/scv/shared/phase246_20261031')
SPEC=[('Irish_026046_2001-03-24','p26_r46.tar.gz','026','046','2001-03-24'),('Irish_035042_2001-08-14','p35_r42.tar.gz','035','042','2001-08-14'),('Irish_039025_2001-07-25','p39_r25.tar.gz','039','025','2001-07-25')]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024**2):h.update(b)
 return h.hexdigest()

def main():
 # One bounded pipeline dependency wait; no scheduled monitor or recurring task.
 epoch=float((ROOT/'START_EPOCH').read_text()) if (ROOT/'START_EPOCH').exists() else time.time()
 while not (ROOT/'ARCHIVE_SUMMARY.json').exists() and time.time()-epoch<1800:
  status=ROOT/'STATUS.json'
  if status.exists() and json.loads(status.read_text()).get('stage')=='failed':break
  time.sleep(5)
 import numpy as np,rasterio
 from rasterio.windows import Window
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
 start=time.monotonic();results=[]
 for scene,filename,path,row,date in SPEC:
  receipt=ROOT/(scene+'_ARCHIVE.json')
  if not receipt.exists():results.append({'scene':scene,'status':'archive_pending'});continue
  a=json.loads(receipt.read_text());members=[m['name'] for m in a['members'] if m['regular_file']]
  # Standard original Landsat7 filenames use B10/B20/B30 or B1/B2/B3.
  band_names={str(b):[n for n in members if re.search(r'(?:_|[.])B'+str(b)+r'0?\.(?:TIF|TIFF)$',n,re.I)] for b in [1,2,3]}
  mask_names=[n for n in members if re.search(r'(?:mask|truth|manual)',Path(n).name,re.I) and Path(n).suffix.lower() in ['.tif','.tiff','.img'] and not re.search(r'(?:qmask|cfmask|fmask|bqa)',Path(n).name,re.I)]
  record={'scene':scene,'RGB_band_order':['3','2','1'],'band_candidates':band_names,'mask_candidates':mask_names,'source_archive_sha256':a['sha256']}
  if any(len(v)!=1 for v in band_names.values()) or len(mask_names)!=1:
   record['status']='archive_member_binding_pending';results.append(record);continue
  selected={b:n[0] for b,n in band_names.items()}|{'reference':mask_names[0]}
  dest=ROOT/(scene+'_native_selected');dest.mkdir(exist_ok=True);paths={}
  with tarfile.open(ROOT/filename,'r:gz') as tar:
   byname={m.name:m for m in tar.getmembers()}
   for key,name in selected.items():
    m=byname[name]
    if not m.isfile() or m.size>512*1024**2:raise RuntimeError('unexpected native member size or type')
    p=dest/Path(name).name
    if not p.exists():
     with tar.extractfile(m) as f,p.with_suffix(p.suffix+'.part').open('wb') as g:shutil.copyfileobj(f,g,1024**2)
     p.with_suffix(p.suffix+'.part').rename(p)
    if p.stat().st_size!=m.size:raise RuntimeError('extracted native member length mismatch')
    paths[key]=p
  sources={k:rasterio.open(p) for k,p in paths.items()}
  try:
   label=sources['reference'];grid=lambda d:{'shape':[d.height,d.width],'count':d.count,'CRS':str(d.crs),'transform':list(d.transform)[:6],'nodata':d.nodata,'dtype':list(d.dtypes)}
   record['grids']={k:grid(d) for k,d in sources.items()};record['native_file_sha256']={k:sha(p) for k,p in paths.items()}
   record['same_grid']=all(d.crs==label.crs and d.transform==label.transform and d.width==label.width and d.height==label.height and d.count==1 for d in sources.values())
   if not record['same_grid'] or label.crs is None:
    record['status']='native_grid_pending';results.append(record);continue
   native=np.zeros(256,dtype=np.int64);joint=np.zeros(256,dtype=np.int64);nonzero_reference_missing_RGB=0;input_valid_count=0;valid_label_mask_disagreement=0
   dn_hist={k:np.zeros(256,dtype=np.int64) for k in ['1','2','3']};bad_dtype=False
   for y in range(0,label.height,512):
    for x in range(0,label.width,512):
     w=Window(x,y,min(512,label.width-x),min(512,label.height-y));ref=label.read(1,window=w)
     rgb=[sources[k].read(1,window=w) for k in ['1','2','3']]
     if ref.min()<0 or ref.max()>255 or any(im.min()<0 or im.max()>255 for im in rgb):bad_dtype=True;break
     ok=np.logical_and.reduce([(im>0)&(sources[k].read_masks(1,window=w)>0) for k,im in zip(['1','2','3'],rgb)])
     native+=np.bincount(ref.ravel(),minlength=256);joint+=np.bincount(ref[ok].ravel(),minlength=256)
     nonzero_reference_missing_RGB+=int(((ref!=0)&~ok).sum());input_valid_count+=int(ok.sum());valid_label_mask_disagreement+=int(((label.read_masks(1,window=w)==0)&(ref!=0)).sum())
     for k,im in zip(['1','2','3'],rgb):dn_hist[k]+=np.bincount(im[ok].ravel(),minlength=256)
    if bad_dtype:break
   values=np.flatnonzero(native).tolist();record['native_label_counts']={str(v):int(native[v]) for v in values};record['joint_label_counts']={str(v):int(joint[v]) for v in values}
   record['joint_classes']={'Surface':int(joint[128]),'Cloud':int(joint[192]+joint[255]),'Shadow':int(joint[64])}
   record['ignored_native_Fill']=int(native[0]);record['reference_valid_but_RGB_missing']=nonzero_reference_missing_RGB;record['input_valid_pixels']=input_valid_count;record['reference_mask_disagreement']=valid_label_mask_disagreement
   record['RGB_DN_histogram']={k:{str(i):int(n) for i,n in enumerate(v) if n} for k,v in dn_hist.items()}
   record['unknown_native_codes']=sorted(set(values)-{0,64,128,192,255});record['DN_beyond_uint8']=bad_dtype
   texts='\n'.join(t['text'] for t in a['texts']);fields={}
   for line in texts.splitlines():
    m=re.match(r'\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$',line)
    if m:fields[m[1]]=m[2].strip('"')
   keep=[k for k in fields if any(s in k for s in ['SUN_ELEVATION','EARTH_SUN_DISTANCE','DATE_ACQUIRED','ACQUISITION_DATE','WRS_PATH','WRS_ROW','RADIANCE','REFLECTANCE','LMAX','LMIN','QCAL','SPACECRAFT_ID','SENSOR_ID'])]
   record['MTL_identity_and_radiometry']={k:fields[k] for k in keep};record['radiometric_conversion_status']='not_converted; check original MTL and pre-collection ESUN requirements before model input'
   path_ok=any(str(int(path))==fields[k].lstrip('0') for k in ['WRS_PATH'] if k in fields)
   row_ok=any(str(int(row))==fields[k].lstrip('0') for k in ['WRS_ROW'] if k in fields)
   date_ok=any(fields.get(k,'')==date for k in ['DATE_ACQUIRED','ACQUISITION_DATE'])
   record['MTL_identity_matches']={'path':path_ok,'row':row_ok,'date':date_ok}
   record['status']='native_grid_and_support_pass' if not record['unknown_native_codes'] and not bad_dtype and all(record['MTL_identity_matches'].values()) and min(record['joint_classes'].values())>0 else 'identity_or_support_pending'
   record['confirmation_ready']=False
   results.append(record)
  finally:
   for d in sources.values():d.close()
 out={'results':results,'new_model_calls':0,'new_training':0,'reference_pixels_read':True,'elapsed_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rasterio':rasterio.__version__,'numpy':np.__version__,'input_validity':'all three native DN>0 and raster masks valid; reference0 ignored; no clipping, resampling or model input normalization'}
 (ROOT/'DATA_ACCEPTANCE.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({r['scene']:r['status'] for r in results}))

if __name__=='__main__':main()
