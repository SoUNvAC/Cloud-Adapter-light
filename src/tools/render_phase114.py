"""Render anonymous evidence from original pixels; no models or labels."""
import json,hashlib,collections,tarfile,time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
from PIL import Image,ImageDraw
from install_phase111_runtime import host_free
R=Path(__file__).resolve().parents[2];B=Path('/home/scv/shared/phase114_20261014');INPUT=Path('/home/scv/shared/phase114_anonymous_inputs_20261014.json');ORIGINAL=Path('/home/scv/shared/data/pixbox_s2_cmix_5036991/phase104_20261013');P112=Path('/home/scv/shared/phase112_20261014');P111=Path('/home/scv/shared/phase111_ocm_runtime')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def crop(a,top,left,size,fill=0):
 out=np.full((size,size),fill,dtype=a.dtype);y0=max(top,0);x0=max(left,0);y1=min(top+size,a.shape[0]);x1=min(left+size,a.shape[1]);out[y0-top:y1-top,x0-left:x1-left]=a[y0:y1,x0:x1];return out

def panel(arrays,nodata,saturated,path,pixel_point,cell_box,resolution,width_m):
 rgb=np.stack(arrays,axis=-1).astype(np.float32)/np.float32(10000);rgb=np.rint(np.clip(rgb/.3,0,1)*255).astype(np.uint8)
 rgb[nodata]=[75,75,75];rgb[saturated]=[230,40,160]
 base=Image.fromarray(rgb).resize((1024,1024),Image.Resampling.NEAREST);canvas=Image.new('RGB',(1024,1088),'white');canvas.paste(base,(0,0));draw=ImageDraw.Draw(canvas);scale=1024/rgb.shape[0]
 x,y=[v*scale for v in pixel_point];box=[v*scale for v in cell_box];draw.rectangle(box,outline=(0,255,255),width=2)
 for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:draw.line((x+dx*5,y+dy*5,x+dx*16,y+dy*16),fill=(255,255,0),width=2)
 length=100 if width_m==640 else 2000;pixels=length/width_m*1024;draw.line((30,995,30+pixels,995),fill='white',width=5);draw.text((30,976),str(length)+' m',fill='white')
 draw.text((12,1030),f'Native {resolution} m; extent {width_m} m; TOA fixed 0-0.3; nearest display',fill='black');draw.text((12,1054),'Yellow: original point. Cyan:20m cell. Gray:no observation/outside. Pink:saturated.',fill='black');canvas.save(path,optimize=True)

def main():
 start=time.monotonic();lock=json.loads((R/'src/research_results/phase114_20261014/SAMPLING_BLIND_LOCK.json').read_text())
 for n,h in lock['source_sha256'].items():assert sha(R/n)==h,n
 assert sha(INPUT)==lock['anonymous_inputs_sha256'];jobs=json.loads(INPUT.read_text())['rows'];assert len(jobs)==lock['n_samples'];assert all(not({'reference','sampling_layer','Omni','group_id'}&set(x)) for x in jobs)
 assert host_free()>=4000000000;B.mkdir(exist_ok=False);assets=B/'assets';assets.mkdir();manifest=json.loads((R/'src/research_results/phase104_20261013/object_manifest.json').read_text());products={p['product_id']:p for p in manifest['products']};dsstatus=json.loads((ORIGINAL/'download_status.json').read_text());b8astatus=json.loads((P112/'DOWNLOAD_STATUS.json').read_text());old=json.loads((P111/'ASSET_VERIFICATION.json').read_text());byproduct=collections.defaultdict(list)
 for j in jobs:byproduct[j['product_id']].append(j)
 geometry=[];input_hashes={};images={}
 for pid,jj in byproduct.items():
  product=products[pid];assert product['quantification']==10000 and product['offset']==0
  paths={b:ORIGINAL/'imagery'/pid/(b+'.jp2') for b in ['B04','B03','B02']}
  paths['B8A']=Path(b8astatus['files'][pid]['path']) if pid in b8astatus['files'] else P111/'assets/B8A.jp2'
  spectral=[];spec_nodata=[];spec_sat=[];rgb_native={}
  for band,path in paths.items():
   expected=dsstatus['files'][pid+'/'+band+'.jp2']['sha256'] if band!='B8A' else b8astatus['files'][pid]['sha256'] if pid in b8astatus['files'] else next(x['sha256'] for x in old['files'] if x['path']==str(path))
   assert sha(path)==expected;input_hashes[pid+'/'+band]=expected
   with rasterio.open(path) as ds:
    g=product['grid20' if band=='B8A' else 'grid10'];assert ds.crs.to_string()==product['crs'];assert (ds.height,ds.width)==(g['NROWS'],g['NCOLS']);assert tuple(ds.transform)[:6]==(g['XDIM'],0,g['ULX'],0,g['YDIM'],g['ULY']);assert ds.dtypes==('uint16',)
    native=ds.read(1)
    if band!='B8A':rgb_native[band]=native
    if band=='B02':continue
    nodata=native==0;sat=native==65535
    if band!='B8A':
     a=ds.read(1,out_shape=(5490,5490),resampling=Resampling.bilinear)
     nodata=binary_dilation(nodata,structure=np.ones((3,3),bool)).reshape(5490,2,5490,2).any(axis=(1,3));sat=binary_dilation(sat,structure=np.ones((3,3),bool)).reshape(5490,2,5490,2).any(axis=(1,3))
    else:a=native
    a=a.copy();a[nodata|sat]=0;spectral.append((band,a));spec_nodata.append(nodata);spec_sat.append(sat);del native
  spec=dict(spectral);nd=np.logical_or.reduce(spec_nodata);sat=np.logical_or.reduce(spec_sat)
  for j in jj:
   g=product['grid20'];fx=(j['native_x_m']-g['ULX'])/20;fy=(g['ULY']-j['native_y_m'])/20;assert int(np.floor(fx))==j['col20'] and int(np.floor(fy))==j['row20'];assert int(np.floor(fx*2))==j['col10'] and int(np.floor(fy*2))==j['row10']
   for width in [640,10240]:
    n=width//20;left=j['col20']-n//2;top=j['row20']-n//2;rgb=[]
    for band in ['B04','B03','B02']:
     rgb.append(crop(rgb_native[band],top*2,left*2,n*2))
    rgbnd=np.logical_or.reduce([a==0 for a in rgb]);rgbsat=np.logical_or.reduce([a==65535 for a in rgb]);false=[crop(spec[b],top,left,n) for b in ['B8A','B04','B03']];fn=crop(nd,top,left,n,fill=True);fs=crop(sat,top,left,n,fill=False)
    pixel=[fx-left,fy-top];box=[j['col20']-left,j['row20']-top,j['col20']-left+1,j['row20']-top+1]
    assert box[0]<=pixel[0]<box[2] and box[1]<=pixel[1]<box[3]
    for kind,aa,nn,ss,point,cell,res in [('RGB',rgb,rgbnd,rgbsat,[v*2 for v in pixel],[v*2 for v in box],10),('FALSE',false,fn,fs,pixel,box,20)]:
     name=j['anonymous_id']+'_'+str(width)+'_'+kind+'.png';panel(aa,nn,ss,assets/name,point,cell,res,width);images[name]=sha(assets/name)
    geometry.append({'anonymous_id':j['anonymous_id'],'extent_m':width,'true_point_inside_original_20m_cell':True,'RGB_shape':[n*2,n*2],'false_shape':[n,n],'same_geographic_bounds':True,'native_RGB_point_not_replaced_by_cell_center':True,'no_observation_RGB_pixels':int(rgbnd.sum()),'saturated_RGB_pixels':int(rgbsat.sum()),'no_observation_false_pixels':int(fn.sum()),'saturated_false_pixels':int(fs.sum())})
   print('Rendered',j['anonymous_id'],flush=True)
  del spec,spectral,nd,sat,spec_nodata,spec_sat,rgb_native
  assert host_free()>=4000000000
 report={'status':'passed','n_samples':len(jobs),'n_images':len(images),'geometry':geometry,'image_sha256':images,'input_hashes':input_hashes,'labels_or_predictions_read':False,'new_forwards':0,'new_fits':0,'new_downloads':0,'display_TOA_range':[0,.3],'false_color_bilinear_full_native_uint16_and_halo_same_as_frozen_input':True,'point_marker':'original native coordinate','seconds':time.monotonic()-start,'host_free_after_bytes':host_free()}
 (B/'RENDER_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
 with tarfile.open(B/'anonymous_evidence.tar.gz','x:gz') as t:t.add(assets,arcname='assets')
 (B/'EVIDENCE_SEAL.json').write_text(json.dumps({'archive_sha256':sha(B/'anonymous_evidence.tar.gz'),'archive_bytes':(B/'anonymous_evidence.tar.gz').stat().st_size,'render_acceptance_sha256':sha(B/'RENDER_ACCEPTANCE.json'),'images':len(images),'blind_page_material_only':True},indent=2)+'\n');print('Anonymous evidence sealed',len(images),flush=True)
if __name__=='__main__':main()
