"""Prepare prediction-masked cases; cache-only audit, no neural library imports."""
from pathlib import Path
import json,os,time,hashlib,re,resource,zipfile
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',GDAL_CACHEMAX='64')
import numpy as np,cv2,rasterio
from rasterio.windows import Window
from PIL import Image,ImageDraw
R=Path('/home/scv/shared/phase248_20261031');P=Path('/home/scv/shared/phase241_20261030')
SCENES=['LC81321192014054LGN00','LC81620582014104LGN00','LC81130632014241LGN00','LC82290572014141LGN00','LC81540102013205LGN00']
MODELS=['Source','C21701','C21702']
def js(p):return json.loads(Path(p).read_text())
def save(n,v):(R/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(4194304):h.update(b)
 return h.hexdigest()
def a_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def cm(y,p):
 v=y!=255;return np.bincount((y[v].astype(np.int64)*3+p[v]).ravel(),minlength=9).reshape(3,3)
def crop(a,row,col,side,fill=0):
 h,w=a.shape[:2];r0=row-side//2;c0=col-side//2;out=np.full((side,side)+a.shape[2:],fill,a.dtype)
 ra=max(r0,0);rb=min(r0+side,h);ca=max(c0,0);cb=min(c0+side,w)
 out[ra-r0:rb-r0,ca-c0:cb-c0]=a[ra:rb,ca:cb];return out
def rgb_view(a,limits):
 lo,hi=limits;return np.uint8(np.clip((a.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)*255)
PALETTE=np.zeros((256,3),np.uint8);PALETTE[0]=[70,155,70];PALETTE[1]=[240,240,240];PALETTE[2]=[105,65,180];PALETTE[255]=[0,0,0]
def panel(a,label,side=220,nearest=False):
 im=Image.fromarray(a).resize((side,side),Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR);out=Image.new('RGB',(side,side+25),'#dddddd');out.paste(im,(0,25));d=ImageDraw.Draw(out);d.text((4,5),label,fill='black');cx=side//2;cy=25+side//2;d.line((cx-7,cy,cx+7,cy),fill='red',width=1);d.line((cx,cy-7,cx,cy+7),fill='red',width=1);return out
def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1800,1800));started=time.monotonic()
 assert not (R/'STAGE_A_READY.json').exists(),'do not overwrite case package'
 E=js(P/'EVAL_INPUTS.json');N=js(P/'NATIVE_PLAN.json');allindex=[json.loads(l) for l in (P/'TEST_SCORE_INDEX.jsonl').read_text().splitlines()]
 idx={(v['scene'],v['name'],v['model']):v for v in allindex if v['scene'] in SCENES}
 hidden=[];technical=[];flow=[];case_a=[];figures=[];rng=np.random.default_rng(24801)
 for si,scene in enumerate(SCENES):
  domain='SPARCS' if si==4 else 'L8';rows=sorted([v for v in E[domain]['test'] if v['scene']==scene],key=lambda v:v['name']);v0=rows[0]
  save('STATUS.json',{'stage':'preparing_prediction_masked_material','scene':scene,'network_calls':0})
  if domain=='L8':
   maskpath=Path(N['test'][scene]['path']);rawpath=maskpath.with_name(scene+'.TIF')
   raw_handle=rasterio.open(rawpath);ref_handle=rasterio.open(maskpath);assert raw_handle.shape==ref_handle.shape and raw_handle.transform==ref_handle.transform and raw_handle.crs==ref_handle.crs
   native=raw_handle.read([4,3,2]).transpose(1,2,0);native_ref=ref_handle.read(1);truth=np.full(native_ref.shape,255,np.uint8);truth[native_ref==128]=0;truth[(native_ref==192)|(native_ref==255)]=1;truth[native_ref==64]=2
   model_truth=truth;mtl=maskpath.with_name(scene+'_MTL.txt');meta=mtl.read_text();georef={'CRS':str(raw_handle.crs),'transform':list(raw_handle.transform)[:6],'native_shape':list(raw_handle.shape),'raw_count':raw_handle.count,'raw_dtype':list(raw_handle.dtypes),'raw_georef_matches_reference':True}
  else:
   rawpath=Path(v0['multispectral_path']);maskpath=Path(v0['mask_path']);native=np.asarray(Image.open(v0['image_path']).convert('RGB'));native_ref=np.asarray(Image.open(maskpath));lut=np.array([2,2,0,0,0,1,0],np.uint8);truth=lut[native_ref];model_truth=cv2.resize(truth,(512,512),interpolation=cv2.INTER_NEAREST);raw_handle=rasterio.open(rawpath);ref_handle=None;mtl=rawpath.with_name(scene+'_mtl.txt');meta=mtl.read_text();georef={'CRS':str(raw_handle.crs),'transform':list(raw_handle.transform)[:6],'native_shape':list(raw_handle.shape),'photo_shape':list(native.shape),'mask_shape':list(native_ref.shape),'raw_count':raw_handle.count,'raw_dtype':list(raw_handle.dtypes),'same_native_dimensions':raw_handle.shape==native_ref.shape==native.shape[:2],'native_label_georef':'PNG has no CRS; same grid of official scene crop not independently located in full L1'}
  shadow=(model_truth==2).astype(np.uint8);interior=cv2.erode(shadow,np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool);boundary=shadow.astype(bool)&~interior
  tiny=native[::max(1,native.shape[0]//512),::max(1,native.shape[1]//512)];limits=np.percentile(tiny,[2,98]);overview=rgb_view(native[::8,::8],limits);Image.fromarray(overview).save(R/f'A_scene{si+1}_overview.png')
  stats={'scene':scene,'domain':domain,'inputs':len(rows),'raw_raster':str(rawpath),'raw_SHA':sha(rawpath),'reference':str(maskpath),'reference_SHA':sha(maskpath),'georef':georef,'mtl_SHA':sha(mtl),'sun':{k:(re.search(r'^\s*'+k+r'\s*=\s*([^\r\n]+)',meta,re.M).group(1).strip() if re.search(r'^\s*'+k+r'\s*=\s*([^\r\n]+)',meta,re.M) else None) for k in ['SUN_AZIMUTH','SUN_ELEVATION']},'input_hash_verified':0,'score_hash_verified':0,'label_cache_exact':0,'rgb_reconstruction_exact':0,'rgb_reconstruction_max_error':0,'rgb_reconstruction_MAE_sum':0.,'axis':'x=row,y=column','overlap_or_duplicate_coordinates':False,'cache_confusion_exact':True}
  chosen_candidates={k:[] for k in range(8)};census={m:{'all':np.zeros((3,3,3),np.int64),'boundary':np.zeros((3,3),np.int64),'interior':np.zeros((3,3),np.int64)} for m in MODELS[1:]};coords=set();nvalid=0;frame_data={}
  for ri,v in enumerate(rows):
   assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256'];stats['input_hash_verified']+=1
   image=np.asarray(Image.open(v['image_path']).convert('RGB'));ann=np.asarray(Image.open(v['mask_path']))
   if domain=='L8':
    rr=int(v['x']);cc=int(v['y']);assert (rr,cc) not in coords;coords.add((rr,cc));assert rr%512==0 and cc%512==0
    assert rr+512<=native.shape[0] and cc+512<=native.shape[1];raw_rgb=native[rr:rr+512,cc:cc+512].astype(np.float32);reconstructed=((raw_rgb-raw_rgb.min())/(raw_rgb.max()-raw_rgb.min()+1e-6)*255).astype(np.uint8)
    assert image.shape==(512,512,3);err=np.abs(reconstructed.astype(np.int16)-image.astype(np.int16));stats['rgb_reconstruction_exact']+=int(not err.any());stats['rgb_reconstruction_max_error']=max(stats['rgb_reconstruction_max_error'],int(err.max()));stats['rgb_reconstruction_MAE_sum']+=float(err.mean())
    y=truth[rr:rr+512,cc:cc+512];bd=boundary[rr:rr+512,cc:cc+512];inside=interior[rr:rr+512,cc:cc+512];expected_ann=np.zeros(y.shape,np.uint8);aa=native_ref[rr:rr+512,cc:cc+512];expected_ann[aa==64]=1;expected_ann[aa==192]=2;expected_ann[aa==255]=3;assert np.array_equal(expected_ann,ann)
   else:
    rr=cc=0;image=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR);y=model_truth;bd=boundary;inside=interior;reconstructed=image;stats['rgb_reconstruction_exact']+=1
   pred=[]
   for m in MODELS:
    z=idx[scene,v['name'],m];sp=P/z['score_file'];yp=P/z['label_file'];assert sha(sp)==z['score_SHA'];stats['score_hash_verified']+=1
    cached_y=np.load(yp,allow_pickle=False);assert np.array_equal(y,cached_y);assert sha(yp)==z['label_SHA'];p=np.load(sp,mmap_mode='r',allow_pickle=False);assert p.dtype==np.float32 and p.shape==(3,512,512);pr=np.asarray(p.argmax(0),np.uint8);assert a_sha(pr)==z['prediction_SHA'];assert cm(y,pr).tolist()==z['confusion'];pred.append(pr);p._mmap.close()
   stats['label_cache_exact']+=1;nvalid+=int((y!=255).sum());agree=(pred[0]==pred[1])&(pred[0]==pred[2]);disagree=~agree
   for j,m in enumerate(MODELS[1:],1):
    valid=y!=255;census[m]['all']+=np.bincount((9*y[valid].astype(np.int64)+3*pred[0][valid]+pred[j][valid]).ravel(),minlength=27).reshape(3,3,3)
    for key,mask in [('boundary',bd),('interior',inside)]:census[m][key]+=np.bincount((3*pred[0][mask].astype(np.int64)+pred[j][mask]).ravel(),minlength=9).reshape(3,3)
   masks=[inside&agree,bd&agree,inside&disagree,bd&disagree,(y==0)&agree,(y==0)&disagree,y!=255,y!=255]
   for k,mask in enumerate(masks):
    candidates=np.flatnonzero(mask)
    if not len(candidates):continue
    picks=rng.choice(candidates,min(6,len(candidates)),replace=False)
    for flat in picks:
     r,c=divmod(int(flat),512);priority=hashlib.sha256(f'24801|{scene}|{k}|{v["name"]}|{flat}'.encode()).hexdigest();chosen_candidates[k].append((priority,ri,r,c))
  stats['valid_cached_reference_pixels']=nvalid;stats['raw_valid_reference_pixels']=int((truth!=255).sum());stats['rgb_reconstruction_MAE_mean']=stats.pop('rgb_reconstruction_MAE_sum')/len(rows);technical.append(stats)
  for m in MODELS[1:]:flow.append({'scene':scene,'domain':domain,'model':m,'reference_Source_C_27':census[m]['all'].tolist(),'Shadow_boundary_Source_C':census[m]['boundary'].tolist(),'Shadow_interior_Source_C':census[m]['interior'].tolist(),'boundary_definition':'3 pixels on model evaluation grid, full-scene erosion before patch extraction for L8','boundary_Shadow':int(census[m]['boundary'].sum()),'interior_Shadow':int(census[m]['interior'].sum())})
  selected=[];case_order=rng.permutation(8)
  for k in range(8):
   found=None
   for _,ri,r,c in sorted(chosen_candidates[k]):
    v=rows[ri];ar=(int(v.get('x',0))+r if domain=='L8' else r);ac=(int(v.get('y',0))+c if domain=='L8' else c)
    if all((ar-s['evaluation_row'])**2+(ac-s['evaluation_col'])**2>=48**2 for s in selected):found=(ri,r,c,ar,ac);break
   if found is None:hidden.append({'scene':scene,'stratum':k,'status':'no_candidate_with_48px_spacing'});continue
   ri,r,c,ar,ac=found;v=rows[ri];name=f'V{si+1}_{int(case_order[k])+1:02d}';selected.append({'evaluation_row':ar,'evaluation_col':ac})
   input_rgb=np.asarray(Image.open(v['image_path']).convert('RGB'));input_rgb=cv2.resize(input_rgb,(512,512),interpolation=cv2.INTER_LINEAR)
   if domain=='L8':nr,nc=ar,ac;local_y=truth[int(v['x']):int(v['x'])+512,int(v['y']):int(v['y'])+512]
   else:nr=int(np.floor((r+.5)*native.shape[0]/512));nc=int(np.floor((c+.5)*native.shape[1]/512));local_y=model_truth
   ctx=rgb_view(crop(native,nr,nc,768),limits);raw_detail=rgb_view(crop(native,nr,nc,192),limits);inp=crop(input_rgb,r,c,192);gt=PALETTE[crop(local_y,r,c,192,255)]
   card=Image.new('RGB',(880,270),'#dddddd');draw=ImageDraw.Draw(card);draw.text((4,2),name+'  GT: green Surface; white Cloud; purple Shadow; black ignore',fill='black')
   for j,(a,label) in enumerate([(ctx,'Local release RGB context 768px'),(raw_detail,'Native RGB detail 192px'),(inp,'Actual model RGB detail 192px'),(gt,'Reference on model grid')]):card.paste(panel(a,label,220,j==3),(j*220,25))
   card.save(R/(name+'_A.png'));record={'case_id':name,'scene':scene,'domain':domain,'input_name':v['name'],'model_row':r,'model_col':c,'evaluation_row':ar,'evaluation_col':ac,'native_display_row':nr,'native_display_col':nc,'reference_center':int(local_y[r,c]),'stageA_image':name+'_A.png','raw_display_percentiles':[float(x) for x in limits]};case_a.append(record)
   hidden.append({**record,'stratum':k,'stratum_description':['Shadow interior agree','Shadow boundary agree','Shadow interior disagree','Shadow boundary disagree','Surface agree','Surface disagree','unconditioned valid spatial control','unconditioned valid spatial control'][k],'input':v,'score_rows':{m:idx[scene,v['name'],m] for m in MODELS},'status':'selected'})
  # Five anonymous scene sheets; no prediction, metric or stratum labels.
  cards=[Image.open(R/(rec['case_id']+'_A.png')) for rec in sorted(case_a,key=lambda z:z['case_id']) if rec['scene']==scene];sheet=Image.new('RGB',(880,270*len(cards)),'white')
  for j,im in enumerate(cards):sheet.paste(im,(0,j*270))
  sheet.save(R/f'A_scene{si+1}_sheet.png');figures.append(f'A_scene{si+1}_sheet.png');raw_handle.close()
  if ref_handle:ref_handle.close()
 save('TECHNICAL_AUDIT.json',technical);save('STAGE_A_CASES.json',sorted(case_a,key=lambda z:z['case_id']));save('HIDDEN_SELECTION_AND_PREDICTION_BINDING.json',hidden);save('STAGE_B_ERROR_FLOWS.json',flow)
 # Keep later-stage metadata outside the review package to preserve ordering.
 with zipfile.ZipFile(R/'stageA_visuals.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for f in figures+[f'A_scene{i+1}_overview.png' for i in range(5)]:z.write(R/f,f)
  z.write(R/'STAGE_A_CASES.json','STAGE_A_CASES.json');z.write(R/'TECHNICAL_AUDIT.json','TECHNICAL_AUDIT.json')
 save('STAGE_A_READY.json',{'ready':True,'cases':len(case_a),'sheets':figures,'package_SHA256':sha(R/'stageA_visuals.zip'),'package_bytes':(R/'stageA_visuals.zip').stat().st_size,'network_calls':0,'seconds':time.monotonic()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'prediction_displayed':False})
 save('STATUS.json',{'stage':'awaiting_prediction_masked_review','network_calls':0})
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  save('FAILURE.json',{'traceback':traceback.format_exc()});raise
