"""CPU-only six frozen training-input provenance audit; never load dataset/exporter."""
from pathlib import Path,PurePosixPath
import os,sys,json,time,hashlib,zipfile,binascii,platform,importlib.metadata,resource,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path('/home/scv/shared/phase228_20261026');B=R/'bundle';START=time.monotonic();LIMIT=600
def guard():
 assert time.monotonic()-START<LIMIT,'CPU audit time budget reached'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3,'CPU memory budget reached'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4194304),b''):guard();h.update(v)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,v):(R/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def version(n):
 try:return importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:return None
def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));assert not (R/'RUN_STARTED.json').exists();save('RUN_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'CPU_seconds_wall_cap':LIMIT,'memory_cap_GiB':8,'model_calls':0,'GPU_used':False});(R/'PID').write_text(str(os.getpid()))
 samples=js(B/'SAMPLES.json');assert len(samples)==6 and [s['dataset'] for s in samples]==['L8']*3+['SPARCS']*3;freeze=js(B/'FREEZE_AUDIT.json');check=[]
 for v in freeze['inputs']:
  p=B/'snapshot'/Path(*PurePosixPath(v['path'].replace('\\','/')).parts);check.append({'path':v['path'],'expected_SHA':v['sha256'],'actual_SHA':sha(p),'match':sha(p)==v['sha256']})
 assert all(v['match'] for v in check) and sha(B/'SAMPLES.json')==freeze['samples_sha256'];save('FROZEN_CODE_AND_INPUT_ACCEPTED.json',{'status':'passed','files':check,'no_dataset_or_exporter_import':True,'runtime_snapshot_notes':'Frozen source retained as text only; scalar candidate operator reproduced below.'})
 versions={'Python':platform.python_version(),'executable':sys.executable,**{n:version(n) for n in ['numpy','Pillow','tifffile','rasterio']}};save('VERSIONS.json',versions)
 try:import numpy as np
 except ImportError:np=None
 try:from PIL import Image
 except ImportError:Image=None
 try:import tifffile
 except ImportError:tifffile=None
 allowed_current={str(Path(s['image_path']).resolve()) for s in samples};allowed_tiffs=set();opens=[];denied=[]
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0]));ss=str(p.resolve());suffix=p.suffix.lower()
  if any(k in ss for k in ['/ann_dir/','_fixedmask.','_BQA.']):denied.append(ss);raise RuntimeError('228 forbids label pixel access')
  if suffix in ['.png','.tif','.tiff','.jpg','.jp2'] and ss.startswith(('/home/scv/shared/data/','/home/scv/Cloud-Adapter-light/phase64_data/','/home/scv/Cloud-Adapter-light/src/data/')):
   if ss not in allowed_current and ss not in allowed_tiffs:denied.append(ss);raise RuntimeError('Not a frozen train input or bound crop')
   opens.append({'path':ss,'kind':'current_train_PNG' if ss in allowed_current else 'bound_original_crop_TIFF'})
 sys.addaudithook(audit);rows=[];inventory=[]
 crop_roots=[Path(v) for v in ['/home/scv/shared/data/l8_biome_crop','/home/scv/Cloud-Adapter-light/src/data/l8_biome_crop','/home/scv/Cloud-Adapter-light/data/l8_biome_crop','/home/scv/Cloud-Adapter-light/phase64_data/l8_biome_crop','/home/scv/shared/data/l8_biome_raw/l8_biome_crop']]
 for sample in samples[:3]:
  guard();p=Path(sample['image_path']);row={**sample,'logical_membership':'frozen_target_train','original_path':str(p),'actual_path':str(p.resolve()),'current_SHA':None,'current_SHA_pass':False,'original_crop_path':None,'original_crop_SHA':None,'binding_evidence':None,'band_order_evidence':'unknown; frozen code assumes indices3/2/1 as B4/B3/B2, not physical proof','physical_B4_B3_B2_confirmed':False,'current_shape':None,'current_dtype':None,'original_shape':None,'original_dtype':None,'grid_metadata':None,'max_abs_diff':None,'mean_abs_diff':None,'different_spatial_pixels':None,'different_channel_values':None,'status':'UNRESOLVED'}
  try:
   if not p.is_file():row.update(status='MISSING_CURRENT',reason='Frozen current PNG absent');rows.append(row);continue
   row['current_SHA']=sha(p);row['current_SHA_pass']=row['current_SHA']==sample['expected_current_image_sha256']
   if not row['current_SHA_pass']:row.update(status='IDENTITY_MISMATCH',reason='Frozen current PNG hash changed; no pixel comparison');rows.append(row);continue
   if Image is None or np is None:row.update(status='DEPENDENCY_MISSING',reason='Pillow/NumPy unavailable; no dependency installation');rows.append(row);continue
   with Image.open(p) as im:row['PNG_mode']=im.mode;current=np.asarray(im.convert('RGB')).copy()
   row.update(current_shape=list(current.shape),current_dtype=str(current.dtype))
   candidates=[];searched=[]
   # Same-name binding only: no coordinate inference or full-scene recropping.
   for root in crop_roots:
    for folder in [root/'img_dir',root/'img_dir/train',root/'img_dir/test',root/'img_dir/val']:
     for ext in ['.tif','.tiff','.TIF','.TIFF']:
      q=folder/(p.stem+ext);searched.append(str(q))
      if q.is_file():candidates.append(q.resolve())
   for ext in ['.tif','.tiff','.TIF','.TIFF']:
    q=p.with_suffix(ext);searched.append(str(q))
    if q.is_file():candidates.append(q.resolve())
   bio=p.name.split('_')[0];scene=Path('/home/scv/shared/data/l8_biome_raw/l8biome')/bio/sample['scene'];whole=[{'path':str(q),'bytes':q.stat().st_size,'not_used_as_crop':True} for q in [scene/(sample['scene']+'.TIF')] if q.is_file()];inventory.append({'sample':sample['name'],'exact_crop_candidates_checked':searched,'unique_matching_crops':[str(q) for q in sorted(set(candidates))],'whole_scene_original_available_not_read':whole})
   candidates=sorted(set(candidates))
   if not candidates:row.update(status='MISSING_ORIGINAL',reason='No same-name cropped TIFF in known crop/current image directories; only whole-scene TIFF found. No recrop or coordinate inference.',whole_scene_available=whole);rows.append(row);continue
   if len(candidates)!=1:row.update(status='UNRESOLVED',reason='Multiple distinct same-name crop originals; no automatic choice');rows.append(row);continue
   q=candidates[0];allowed_tiffs.add(str(q));row.update(original_crop_path=str(q),original_crop_SHA=sha(q),binding_evidence='Same basename preserved by frozen create_l8.py and crop img_dir naming; candidate TIFF unique')
   if tifffile is None:row.update(status='DEPENDENCY_MISSING',reason='tifffile unavailable; no install');rows.append(row);continue
   with tifffile.TiffFile(q) as tf:
    series=tf.series[0];shape=series.shape;axes=series.axes;page=tf.pages[0];spp=int(page.samplesperpixel);row.update(original_shape=list(shape),original_dtype=str(series.dtype),TIFF_axes=axes,TIFF_samples_per_pixel=spp);row['grid_metadata']={n:str(page.tags[n].value)[:16000] for n in ['ModelPixelScaleTag','ModelTiepointTag','ModelTransformationTag','GeoKeyDirectoryTag','GeoAsciiParamsTag'] if n in page.tags};description=page.description or '';row['TIFF_description']=description[:16000]
    if len(shape)!=3 or tuple(shape[:2])!=tuple(current.shape[:2]) or shape[2]<4 or spp!=shape[2] or not axes.startswith('YX'):row.update(status='UNRESOLVED',reason='Original not clearly HWC multiband crop on current PNG grid; no transpose/resampling');rows.append(row);continue
    original=series.asarray()
   if not np.isfinite(original).all():row.update(status='UNRESOLVED',reason='Original contains nonfinite values; no repair');rows.append(row);continue
   rgb=np.zeros((original.shape[0],original.shape[1],3),dtype=np.float32);rgb[:,:,0]=original[:,:,3];rgb[:,:,1]=original[:,:,2];rgb[:,:,2]=original[:,:,1];rgb=(rgb-rgb.min())/(rgb.max()-rgb.min()+1e-6);candidate=(rgb*255).astype(np.uint8);diff=np.abs(candidate.astype(np.int16)-current.astype(np.int16));row.update(max_abs_diff=int(diff.max()),mean_abs_diff=float(diff.mean()),different_spatial_pixels=int(np.any(diff!=0,axis=2).sum()),different_channel_values=int(np.count_nonzero(diff)),status='EXACT_PIXEL_MATCH' if np.array_equal(candidate,current) else 'PIXEL_MISMATCH',comparison_scope='one frozen index operator only; true physical band identities unknown unless independently recorded; PNG encoding bytes not compared',candidate_decoded_RGB_SHA=hashlib.sha256(candidate.tobytes()).hexdigest(),current_decoded_RGB_SHA=hashlib.sha256(current.tobytes()).hexdigest())
  except Exception as e:row.update(status='UNRESOLVED',reason=repr(e))
  rows.append(row);save('PARTIAL_ROWS.json',rows)
 save('L8_ORIGINAL_LOOKUP.json',inventory)
 paths=[Path('/home/scv/shared/data/sparcs/l8cloudmasks.zip'),Path('/home/scv/Cloud-Adapter-light/phase64_data/sparcs/l8cloudmasks.full.zip'),Path('/home/scv/Cloud-Adapter-light/phase64_data/sparcs/l8cloudmasks.zip')];zip_inventory=[{'path':str(p),'bytes':p.stat().st_size if p.exists() else None} for p in paths];archive=next((p for p in paths if p.is_file() and p.stat().st_size==1555733976),None);ident={'candidates':zip_inventory,'path':str(archive) if archive else None,'expected_bytes':1555733976,'expected_SHA':'5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a','status':'MISSING_ARCHIVE'};read_members=[];notes=[]
 if archive:
  ident.update(bytes=archive.stat().st_size,SHA=sha(archive));ident['status']='VERIFIED' if ident['SHA']==ident['expected_SHA'] else 'IDENTITY_MISMATCH'
 save('SPARCS_ARCHIVE_IDENTITY.json',ident)
 if ident['status']=='VERIFIED':
  with zipfile.ZipFile(archive) as z:
   for v in z.infolist():
    if not v.is_dir() and 'readme' in PurePosixPath(v.filename).name.lower() and v.file_size<=131072:
     text=z.read(v).decode('utf-8',errors='replace');notes.append({'member':v.filename,'text':text,'bytes':v.file_size});read_members.append({'member':v.filename,'kind':'bundled_readme_text'})
   for sample in samples[3:]:
    guard();p=Path(sample['image_path']);row={**sample,'logical_membership':'frozen_target_train','original_path':str(p),'actual_path':str(p.resolve()),'current_SHA':sha(p) if p.is_file() else None,'archive_identity':ident['status'],'archive_path':str(archive),'archive_SHA':ident['SHA'],'member_path':None,'member_matches':0,'member_SHA':None,'CRC_expected':None,'CRC_actual':None,'CRC_pass':False,'byte_equal':None,'official_photo_generation_rule':'unknown; BYTE_MATCH proves release identity, not band/stretch/gamma recipe','status':'UNRESOLVED'}
    matches=[v for v in z.infolist() if not v.is_dir() and PurePosixPath(v.filename).name==sample['name']];row['member_matches']=len(matches)
    if len(matches)!=1:row.update(status='MISSING_MEMBER' if not matches else 'UNRESOLVED',reason='Frozen photo member absent or not unique');rows.append(row);continue
    member=matches[0];row.update(member_path=member.filename,CRC_expected=f'{member.CRC:08x}',member_bytes=member.file_size);h=hashlib.sha256();crc=0;total=0;equal=p.is_file();current=p.open('rb') if p.is_file() else None
    try:
     with z.open(member) as f:
      for b in iter(lambda:f.read(1048576),b''):
       guard();h.update(b);crc=binascii.crc32(b,crc);total+=len(b)
       if current is not None and current.read(len(b))!=b:equal=False
     if current is not None and current.read(1):equal=False
    finally:
     if current is not None:current.close()
    read_members.append({'member':member.filename,'kind':'frozen_photo','bytes':total});row.update(member_SHA=h.hexdigest(),CRC_actual=f'{crc&0xffffffff:08x}',CRC_pass=(crc&0xffffffff)==member.CRC and total==member.file_size,byte_equal=equal if p.is_file() else None,status=('BYTE_MATCH' if equal else 'BYTE_MISMATCH') if p.is_file() else 'MISSING_CURRENT');rows.append(row)
 else:
  for sample in samples[3:]:
   p=Path(sample['image_path']);rows.append({**sample,'original_path':str(p),'actual_path':str(p.resolve()),'current_SHA':sha(p) if p.is_file() else None,'archive_identity':ident['status'],'member_path':None,'member_matches':None,'member_SHA':None,'CRC_pass':None,'byte_equal':None,'official_photo_generation_rule':'unknown','status':ident['status'],'reason':'No member pixel/data read because archive identity unavailable'})
 save('BUNDLED_README.json',notes);save('RESULT_ROWS.json',rows);save('ACCESS_AUDIT.json',{'opened_data_files':opens,'denied':denied,'ZIP_members_opened':read_members,'model_calls':0,'GPU_imports':False,'label_pixel_reads':0,'val_test_pixel_reads':0,'physical_test_directory_exception':'One L8 sample resides in a legacy test folder but is frozen target_train; only frozen membership determines permission','full_scene_pixel_reads':0,'TIFF_reprojection_resampling_or_coordinate_guessing':False,'bulk_extraction':False,'new_downloads':0,'formula_search':False});guard();assert len(rows)==6 and not denied
 save('COMPLETE.json',{'status':'complete','samples':6,'model_calls':0,'GPU_used':False,'elapsed_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'new_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'script_SHA':sha(Path(__file__))});(R/'EXIT_CODE').write_text('0\n');print(json.dumps({'status':'complete','rows':[(v['dataset'],v['name'],v['status']) for v in rows],'archive_identity':ident['status'],'seconds':time.monotonic()-START}))
if __name__=='__main__':
 try:main()
 except BaseException:(R/'FAILURE.txt').write_text(traceback.format_exc());(R/'EXIT_CODE').write_text('1\n');raise
