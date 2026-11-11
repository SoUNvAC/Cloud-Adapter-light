"""Complete fixed-scene native-version comparison and prepare one locked input recipe."""
from pathlib import Path
import os,time,json,tarfile,re,hashlib,shutil,resource,math
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path('/home/scv/shared/phase246_20261031/integrated_finish_2015')
PREVIOUS=ROOT.parent
SCENES=['Irish_026046_2001-03-24','Irish_035042_2001-08-14','Irish_039025_2001-07-25']

def save(name,value):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(t,p)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024**2):h.update(b)
 return h.hexdigest()

def main():
 epoch=(ROOT/'TASK_PID').stat().st_mtime
 while not (ROOT/'ARCHIVE_SUMMARY.json').exists() and time.time()-epoch<1800:
  if (ROOT/'STATUS.json').exists() and json.loads((ROOT/'STATUS.json').read_text()).get('stage')=='failed':break
  time.sleep(5)
 import numpy as np,rasterio
 from rasterio.windows import Window
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
 started=time.monotonic();results=[];tiles=[];luts=np.full(256,255,dtype='uint8');luts[64]=2;luts[128]=0;luts[192]=1;luts[255]=1
 for scene in SCENES:
  save('ANALYSIS_STATUS.json',{'scene':scene,'stage':'preparing_native_pair','new_model_calls':0})
  receipt=ROOT/(scene+'_ARCHIVE.json')
  if not receipt.exists():results.append({'scene':scene,'status':'old_archive_unavailable'});continue
  a=json.loads(receipt.read_text());members=[m['name'] for m in a['members'] if m['regular_file']]
  mask=[n for n in members if 'newmask2015' in Path(n).name and n.lower().endswith('.tif')]
  bands={str(b):[n for n in members if re.search('_B'+str(b)+r'0?\.TIF$',n,re.I)] for b in [1,2,3]}
  if len(mask)!=1 or any(len(v)!=1 for v in bands.values()):results.append({'scene':scene,'status':'old_member_binding_pending','mask_candidates':mask,'band_candidates':bands});continue
  bind={b:v[0] for b,v in bands.items()}|{'mask':mask[0]};dest=ROOT/(scene+'_native');dest.mkdir(exist_ok=True);paths={}
  with tarfile.open(a['archive'],'r:gz') as t:
   for key,name in bind.items():
    p=dest/Path(name).name
    if not p.exists():
     m=t.getmember(name)
     if not m.isfile() or m.size>512*1024**2:raise RuntimeError('unexpected native member size/type')
     with t.extractfile(m) as f,p.with_suffix(p.suffix+'.part').open('wb') as g:shutil.copyfileobj(f,g,1024**2)
     p.with_suffix(p.suffix+'.part').rename(p)
    paths[key]=p
  sources={k:rasterio.open(p) for k,p in paths.items()};old=sources['mask'];previous_mask=list((PREVIOUS/(scene+'_native_selected')).glob('*mask2019.TIF'))
  record={'scene':scene,'old_archive_sha256':a['sha256'],'native_paths':{k:str(p) for k,p in paths.items()},'native_sha256':{k:sha(p) for k,p in paths.items()},'shape':[old.height,old.width],'CRS':str(old.crs),'transform':list(old.transform)[:6]}
  same=all(s.crs==old.crs and s.transform==old.transform and s.width==old.width and s.height==old.height and s.count==1 for s in sources.values())
  record['same_native_grid']=same
  if not same or old.crs is None:
   record['status']='old_grid_pending';results.append(record)
   for s in sources.values():s.close()
   continue
  previous=rasterio.open(previous_mask[0]) if len(previous_mask)==1 else None
  comparable=previous is not None and previous.shape==old.shape and previous.transform==old.transform and previous.crs==old.crs
  native=np.zeros(256,dtype=np.int64);joint=np.zeros(256,dtype=np.int64);cross=np.zeros((256,256),dtype=np.int64);mapped_counts=np.zeros(256,dtype=np.int64);tile_counts=np.zeros(256,dtype=np.int64)
  prepared=ROOT/(scene+'_prepared');prepared.mkdir(exist_ok=True);rgbfile=prepared/'RGB_native_DN.tif';labelfile=prepared/'reference_3class.tif'
  if rgbfile.exists() or labelfile.exists():raise RuntimeError('prepared outputs exist; do not silently overwrite')
  profile=old.profile.copy();profile.update(driver='GTiff',dtype='uint8',count=3,nodata=0,compress='deflate',tiled=True,blockxsize=512,blockysize=512)
  lp=profile|{'count':1,'nodata':255}
  with rasterio.open(rgbfile,'w',**profile) as dst,rasterio.open(labelfile,'w',**lp) as lbl:
   dst.colorinterp=(rasterio.enums.ColorInterp.red,rasterio.enums.ColorInterp.green,rasterio.enums.ColorInterp.blue)
   for y in range(0,old.height,512):
    for x in range(0,old.width,512):
     h=min(512,old.height-y);w=min(512,old.width-x);win=Window(x,y,w,h);ref=old.read(1,window=win);rgb=np.stack([sources[k].read(1,window=win) for k in ['3','2','1']]);valid=(rgb>0).all(0)
     native+=np.bincount(ref.ravel(),minlength=256);joint+=np.bincount(ref[valid].ravel(),minlength=256)
     mapped=luts[ref];mapped[~valid]=255;mapped_counts+=np.bincount(mapped.ravel(),minlength=256)
     dst.write(rgb,window=win);lbl.write(mapped,1,window=win)
     if comparable:
      newer=previous.read(1,window=win);cross+=np.bincount((ref[valid].astype(np.int32)*256+newer[valid]).ravel(),minlength=65536).reshape(256,256)
     # All native grid windows retained; no selection by class or model score.
     tiles.append({'scene':scene,'x':x,'y':y,'width':w,'height':h,'pad_to':512,'RGB':str(rgbfile),'reference':str(labelfile),'valid_reference':int((mapped!=255).sum())})
     tile_counts+=np.bincount(mapped.ravel(),minlength=256)
  for s in sources.values():s.close()
  if previous is not None:previous.close()
  codes=np.flatnonzero(native).tolist();unknown=sorted(set(codes)-{0,64,128,192,255});classes={'Surface':int(mapped_counts[0]),'Cloud':int(mapped_counts[1]),'Shadow':int(mapped_counts[2])}
  transitions=[{'old2015_code':int(i),'new2019_code':int(j),'pixels':int(cross[i,j])} for i,j in zip(*np.nonzero(cross))]
  (prepared/'VERSION_CROSS_COUNTS.json').write_text(json.dumps(transitions,indent=2))
  previous_a=json.loads((PREVIOUS/(scene+'_ARCHIVE.json')).read_text());previous_hashes={Path(n['name']).name:n for n in previous_a['members']}
  old_mtl='\n'.join(t['text'] for t in a['texts']);fields={m.group(1):m.group(2).strip().strip('"') for m in re.finditer(r'^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$',old_mtl,re.M)}
  # Some original members are extensionless; binding/date also remain traceable in original archive names.
  wrs=scene.split('_')[1];date=scene.split('_')[2];identity={'path':fields.get('WRS_PATH')==str(int(wrs[:3])),'row':fields.get('WRS_ROW',fields.get('STARTING_ROW'))==str(int(wrs[3:])),'date':fields.get('DATE_ACQUIRED',fields.get('ACQUISITION_DATE'))==date}
  record.update({'old2015_native_codes':{str(i):int(native[i]) for i in codes},'joint_native_codes':{str(i):int(joint[i]) for i in codes},'mapped_class_counts':classes,'ignored_pixels':int(mapped_counts[255]),'unknown_codes':unknown,'previous_mask_same_grid':comparable,'version_contingency':transitions,'identity_from_old_MTL':identity,
   'prepared_files':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [rgbfile,labelfile]},'tile_count':sum(t['scene']==scene for t in tiles),'count_reconciliation':bool(np.array_equal(tile_counts,mapped_counts)),'status':'prepared' if not unknown and min(classes.values())>0 and all(identity.values()) else 'mapping_or_identity_pending'})
  results.append(record)
 recipe={'RGB_bands':['B30','B20','B10'],'native_data':'original 2015 counterpart, no image/label reprojection','model_window':512,'stride':512,'partial_edges':'zero-pad RGB; ignore255-pad reference','RGB_to_model_uint8':'per-window RGB global minmax: ((RGB.float32-min)/(max-min+1e-6)*255).astype(uint8); source repository L8BiomeCrop/create_l8 convention; no percentile or label-dependent scaling','native_to_reference':{'0':255,'64':2,'128':0,'192':1,'255':1},'input_invalid':'any native RGB DN==0 -> reference ignore255','selection':'all grid windows, no class or score filtering','TOA':'not needed for this locked display-RGB recipe; original radiometric metadata retained, no TOA array substituted','recalibration_or_training':False}
 save('INPUT_RECIPE.json',recipe);save('TILE_INDEX.json',tiles)
 out={'results':results,'prepared_scenes':sum(r['status']=='prepared' for r in results),'new_model_calls':0,'new_training':0,'new_thresholds':0,'analysis_seconds':time.monotonic()-started,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'preparation_recipe':recipe,'label_mapping_basis':'USGS 2015 newmask2015 + TorchGeo published loader; current mask2019 never relabeled by guess','remaining_scope':'three fixed scenes only, not 24-group confirmation; label accuracy not independently certified'}
 save('FINAL_DATA_READINESS.json',out);save('ANALYSIS_STATUS.json',{'stage':'complete','prepared_scenes':out['prepared_scenes'],'new_model_calls':0})

if __name__=='__main__':
 try:main()
 except Exception as e:save('ANALYSIS_STATUS.json',{'stage':'failed','error':repr(e)});raise
