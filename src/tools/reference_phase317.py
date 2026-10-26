"""Independent CPU-only reference construction after every endpoint/evaluation is sealed."""
import io,zipfile,resource
import numpy as np
from PIL import Image
import cv2
from phase317_common import *
def main():
 resource_check();assert 'torch' not in sys.modules and load(ROOT/'ALL_SEALED.json')['status']=='sealed';assert not (ROOT/'REFERENCE_LOCK.json').exists()
 pins={};out=[];start=time.monotonic();ref=ROOT/'reference';ref.mkdir()
 def pin(p):pins[str(p)]=sha(p)
 # CMC: preserve the exact historical common valid support, not a newly selected image-valid mask.
 old300=Path('/home/scv/shared/phase300_20261020');sp=load(old300/'EXPECTED_SUPPORT.json');plan=load(old300/'EVALUATION_INPUT_PLAN.json');pin(old300/'EXPECTED_SUPPORT.json');pin(old300/'EVALUATION_INPUT_PLAN.json')
 support={r['product_id']:r for r in sp['rows'] if r['pool']=='Catalogue'};plans={r['product_id']:r for r in plan['rows'] if r['pool']=='Catalogue'}
 mp=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/masks.zip');pin(mp)
 with zipfile.ZipFile(mp) as archive:
  for row in inputs('CMC')['val']:
   name=row['product'];p=Path('/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue')/(name+'.npz');pin(p)
   with np.load(p) as z:ids=z['flat_indices'];v=z['valid']
   assert ah(ids)==support[name]['flat_indices_sha256'] and ah(v)==support[name]['common_valid_sha256'];valid=np.zeros((1022,1022),bool);valid.ravel()[ids[v]]=True
   raw=archive.read('masks/'+name+'.npy');assert hashlib.sha256(raw).hexdigest()==plans[name]['group']['mask_member_sha256'];mask=np.load(io.BytesIO(raw));assert mask.shape==(1022,1022,3) and (mask.sum(-1)[valid]==1).all();y=mask.argmax(-1).astype(np.uint8)
   p=ref/('CMC_'+name+'.npz');np.savez_compressed(p,truth=y,native=y,valid=valid);out.append(dict(pool='CMC',name=name,scene=row['group_id'],biome='NA',file=str(p),sha256=sha(p),support=int(valid.sum()),invalid=int((~valid).sum())))
 # SPARCS: original 7-class mask indices, fixed512 nearest as preregistered for this round.
 for row in inputs('SPARCS')['val']:
  p=Path(row['mask_path']);pin(p);native=np.asarray(Image.open(p));assert native.ndim==2 and np.isin(native,np.arange(7)).all();native=cv2.resize(native,(512,512),interpolation=cv2.INTER_NEAREST)
  truth=np.array([2,2,0,0,0,1,0],np.uint8)[native];valid=np.ones((512,512),bool);p=ref/('SPARCS_'+row['name']+'.npz');np.savez_compressed(p,truth=truth,native=native,valid=valid);out.append(dict(pool='SPARCS',name=row['name'],scene=row['scene'],biome='NA',file=str(p),sha256=sha(p),support=int(valid.sum()),invalid=0))
 # L8 PNG converter defaulted rawfill0 to clear0. Restore rawfill only in evaluation.
 import rasterio
 from rasterio.windows import Window
 meta=load(ROOT/'REFERENCE_METADATA.json');rows=inputs('L8')['val'];handles={};candidate={k:True for k in ['x_col_y_row','x_row_y_col']};look=[];lut=np.zeros(256,np.uint8);lut[64]=1;lut[192]=2;lut[255]=3
 for scene,m in meta['l8_raw'].items():pin(m['path']);handles[scene]=rasterio.open(m['path'])
 try:
  for row in rows:
   pin(row['mask_path']);png=np.asarray(Image.open(row['mask_path']));assert png.shape==(512,512) and np.isin(png,[0,1,2,3]).all();x,y=int(row['x']),int(row['y']);h=handles[row['scene']];matches={}
   for key,(col,rr) in [('x_col_y_row',(x,y)),('x_row_y_col',(y,x))]:
    raw=h.read(1,window=Window(col,rr,512,512),boundless=True,fill_value=0);assert np.isin(raw,[0,64,128,192,255]).all();matches[key]=bool(np.array_equal(lut[raw],png));candidate[key]&=matches[key]
   look.append(dict(name=row['name'],**matches))
  orientations=[k for k,v in candidate.items() if v];assert orientations,'Neither global axis orientation exactly reproduces all963 originalPNG masks; preserve failure, no fallbackreference'
  chosen=orientations[0]
  for row in rows:
   x,y=int(row['x']),int(row['y']);col,rr=(x,y) if chosen=='x_col_y_row' else (y,x);raw=handles[row['scene']].read(1,window=Window(col,rr,512,512),boundless=True,fill_value=0)
   if len(orientations)>1:
    other=handles[row['scene']].read(1,window=Window(y,x,512,512),boundless=True,fill_value=0);assert np.array_equal(raw,other),'ambiguous rawfill despite identicalPNG; stop'
   valid=raw!=0;truth=np.full(raw.shape,255,np.uint8);truth[raw==128]=0;truth[np.isin(raw,[192,255])]=1;truth[raw==64]=2;p=ref/('L8_'+row['name']+'.npz');np.savez_compressed(p,truth=truth,native=raw,valid=valid)
   out.append(dict(pool='L8',name=row['name'],scene=row['scene'],biome=row['biome'],file=str(p),sha256=sha(p),support=int(valid.sum()),invalid=int((~valid).sum())))
 finally:
  for h in handles.values():h.close()
 table(ROOT/'L8_REFERENCE_ALIGNMENT.csv',look);write(ROOT/'REFERENCE_INPUT_SHA256.json',pins);write(ROOT/'REFERENCE_LOCK.json',dict(status='sealed',rows=out,L8_axes=chosen,L8_all_PNG_reproduced_exactly=True,L8_fill_ignored=True,allfive_training_before_truth=True,epoch=time.time(),resources=resource_check(),seconds=time.monotonic()-start))
 print('Phase317 references verified; rawL8 fill excluded afterexactPNG alignment; allfive training alreadysealed')
if __name__=='__main__':main()
