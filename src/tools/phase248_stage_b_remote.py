"""Reveal cached predictions only after a sealed prediction-masked review."""
from pathlib import Path
import os,json,hashlib,time,zipfile,resource
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',GDAL_CACHEMAX='64')
import numpy as np,cv2,rasterio
from PIL import Image,ImageDraw
R=Path('/home/scv/shared/phase248_20261031');P=Path('/home/scv/shared/phase241_20261030')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def crop(a,r,c,size=192,fill=0):
 h,w=a.shape[:2];out=np.full((size,size)+a.shape[2:],fill,a.dtype);r0=r-size//2;c0=c-size//2;ra=max(r0,0);rb=min(r0+size,h);ca=max(c0,0);cb=min(c0+size,w);out[ra-r0:rb-r0,ca-c0:cb-c0]=a[ra:rb,ca:cb];return out
def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1800,1800));start=time.monotonic()
 assert sha(R/'STAGE_A_REVIEW_LOCKED.json')==js(R/'STAGE_A_REVIEW_SEAL.json')['sha256'];assert not (R/'STAGE_B_READY.json').exists()
 hidden=js(R/'HIDDEN_SELECTION_AND_PREDICTION_BINDING.json');review=js(R/'STAGE_A_REVIEW_LOCKED.json');assert {r['case_id'] for r in review['cases']}=={r['case_id'] for r in hidden if r['status']=='selected'}
 palette=np.zeros((256,3),np.uint8);palette[0]=[70,155,70];palette[1]=[240,240,240];palette[2]=[105,65,180]
 cases=[];sheets=[]
 for si in range(1,6):
  selected=sorted([r for r in hidden if r['status']=='selected' and r['case_id'].startswith('V'+str(si)+'_')],key=lambda r:r['case_id']);cards=[]
  for rec in selected:
   v=rec['input'];rgb=np.asarray(Image.open(v['image_path']).convert('RGB'));rgb=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR);r=rec['model_row'];c=rec['model_col'];row={'case_id':rec['case_id'],'scene':rec['scene'],'stratum':rec['stratum_description'],'reference_center':rec['reference_center'],'model_centers':{}};imgs=[crop(rgb,r,c)];labels=['Actual model RGB']
   y=np.load(P/rec['score_rows']['Source']['label_file'],allow_pickle=False);imgs.append(palette[crop(y,r,c,fill=255)]);labels.append('Reference')
   for m,z in rec['score_rows'].items():
    p=np.load(P/z['score_file'],mmap_mode='r',allow_pickle=False);pred=np.asarray(p.argmax(0),np.uint8);imgs.append(palette[crop(pred,r,c,fill=255)]);labels.append(m);row['model_centers'][m]={'class':int(pred[r,c]),'p':p[:,r,c].tolist()};p._mmap.close()
   card=Image.new('RGB',(900,210),'#dddddd');draw=ImageDraw.Draw(card);draw.text((3,2),rec['case_id']+' green=Surface white=Cloud purple=Shadow black=outside/ignore',fill='black')
   for j,(im,label) in enumerate(zip(imgs,labels)):
    im=Image.fromarray(im).resize((180,180),Image.Resampling.BILINEAR if j==0 else Image.Resampling.NEAREST);card.paste(im,(180*j,30));draw.text((180*j+3,16),label,fill='black');cx=180*j+90;cy=120;draw.line((cx-6,cy,cx+6,cy),fill='red');draw.line((cx,cy-6,cx,cy+6),fill='red')
   cards.append(card);cases.append(row)
  sheet=Image.new('RGB',(900,210*len(cards)),'white')
  for j,card in enumerate(cards):sheet.paste(card,(0,210*j))
  name=f'B_scene{si}_sheet.png';sheet.save(R/name);sheets.append(name)
 # Quantify the existing RGB normalization, without changing any input or fitting signals.
 E=js(P/'EVAL_INPUTS.json');N=js(P/'NATIVE_PLAN.json');norm=[]
 for scene in list(dict.fromkeys(r['scene'] for r in cases)):
  if scene==list(dict.fromkeys(r['scene'] for r in cases))[-1]:continue
  rawpath=Path(N['test'][scene]['path']).with_name(scene+'.TIF')
  with rasterio.open(rawpath) as f:rgb=f.read([4,3,2])
  for v in E['L8']['test']:
   if v['scene']!=scene:continue
   r=int(v['x']);c=int(v['y']);a=rgb[:,r:r+512,c:c+512];lo=int(a.min());hi=int(a.max());span=hi-lo
   norm.append({'scene':scene,'input':v['name'],'min_DN':lo,'max_DN':hi,'span_DN':span,'normalization_gain':255/(span+1e-6),'zero_DN_fraction':float((a==0).mean())})
 files={'STAGE_B_CASES.json':cases,'INPUT_NORMALIZATION_AUDIT.json':norm}
 for name,v in files.items():(R/name).write_text(json.dumps(v,indent=2))
 with zipfile.ZipFile(R/'stageB_visuals.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in sheets+list(files)+['STAGE_B_ERROR_FLOWS.json','HIDDEN_SELECTION_AND_PREDICTION_BINDING.json']:z.write(R/name,name)
 receipt={'ready':True,'cases':len(cases),'prediction_masked_review_SHA':sha(R/'STAGE_A_REVIEW_LOCKED.json'),'package_SHA256':sha(R/'stageB_visuals.zip'),'package_bytes':(R/'stageB_visuals.zip').stat().st_size,'network_calls':0,'seconds':time.monotonic()-start}
 (R/'STAGE_B_READY.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
