"""Post-seal fit selector audit. No model imports; no new L8 fit registration."""
import io,zipfile
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2
from phase318_common import *
from phase318_selectors import product,selection,crop
def main():
 resource_check();assert 'torch' not in sys.modules and load(ROOT/'ALL_SEALED.json')['status']=='sealed' and not (ROOT/'SELECTOR_QUALITY.csv').exists()
 out=[];gaps=[];pins={};totals=defaultdict(lambda:np.zeros(5,np.int64))
 bindings=load('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json')['bindings'];bindings={r['product']:r for r in bindings}
 mp=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/masks.zip');pins[str(mp)]=sha(mp);pins['/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json']=sha('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json')
 def record(pool,n,mode,selector,selected,y,valid,attempt=0):
  support=selected&valid;cc=np.array([int((support&(y==c)).sum()) for c in range(3)]+[int(selected.sum()),int((selected&~valid).sum())],np.int64)
  out.append(dict(pool=pool,product=n,mode=mode,selector=selector,attempt=attempt,true_Surface=int(cc[0]),true_Cloud=int(cc[1]),true_Shadow=int(cc[2]),requested=int(cc[3]),invalid_or_uncovered=int(cc[4]),reference_support=int(cc[:3].sum()),Shadow_precision_pct=100*int(cc[2])/int(cc[:3].sum()) if cc[:3].sum() else None,note='same-crop exposure is paired comparison; unique Flip and exposure Conf have different denominators'))
  totals[pool,mode,selector]+=cc
 with zipfile.ZipFile(mp) as archive:
  for pool in ['CMC','SPARCS','L8']:
   steps=schedule(pool)
   for i,row in enumerate(inputs(pool)['fit']):
    n=name(row);p,q,g,ids=product(pool,i)
    if pool=='L8':
     gaps.append(dict(pool=pool,product=n,scene=row['scene'],requested_shadow=int((p==2).sum()),flip_selected=int(g.sum()),covered=0,reason='Phase317 verified rawfill coordinates only for disjoint val scenes; fit rawfill/orientation not previously verified. No new registration or guessed clear/fill truth. Full val evaluation retains verified317 support.'))
     y=np.full(p.shape,255,np.uint8);valid=np.zeros(p.shape,bool)
    elif pool=='CMC':
     raw=archive.read('masks/'+n+'.npy');assert hashlib.sha256(raw).hexdigest()==bindings[n]['mask_member_sha256'];m=np.load(io.BytesIO(raw),allow_pickle=False);assert m.shape==(*p.shape,3);valid=m.sum(-1)==1;y=m.argmax(-1).astype(np.uint8)
    else:
     path=Path(row['mask_path']);pins[str(path)]=sha(path);raw=np.asarray(Image.open(path));assert np.isin(raw,np.arange(7)).all();raw=cv2.resize(raw,(512,512),interpolation=cv2.INTER_NEAREST);y=np.array([2,2,0,0,0,1,0],np.uint8)[raw];valid=np.ones(p.shape,bool)
    sh=p==2
    for label,chosen in [('all_A_Sh',sh),('Flip_selected',g),('Flip_unselected',sh&~g)]:record(pool,n,'unique_pixels',label,chosen,y,valid)
    for st in (s for s in steps if s['fit_index']==i):
     pp,fg,cg=selection(pool,st);yy=crop(y,pool,st);vv=crop(valid,pool,st);assert int(fg.sum())==int(cg.sum())
     for label,chosen in [('all_A_Sh',pp==2),('Flip_selected',fg),('Flip_unselected',(pp==2)&~fg),('Conf_selected',cg),('Conf_unselected',(pp==2)&~cg)]:record(pool,n,'scheduled_crop_exposure',label,chosen,yy,vv,st['attempt'])
   product.cache_clear()
 for (pool,mode,selector),cc in totals.items():out.append(dict(pool=pool,product='POOLED',mode=mode,selector=selector,attempt=-1,true_Surface=int(cc[0]),true_Cloud=int(cc[1]),true_Shadow=int(cc[2]),requested=int(cc[3]),invalid_or_uncovered=int(cc[4]),reference_support=int(cc[:3].sum()),Shadow_precision_pct=100*int(cc[2])/int(cc[:3].sum()) if cc[:3].sum() else None,note='L8 fit reference coverage unavailable as preregistered; do not infer selector quality there' if pool=='L8' else 'paired scheduled exposure for Flip-vs-Conf'))
 for pool in ['CMC','SPARCS','L8']:
  for mode in ['unique_pixels','scheduled_crop_exposure']:
   a=totals[pool,mode,'all_A_Sh'];f=totals[pool,mode,'Flip_selected'];u=totals[pool,mode,'Flip_unselected'];assert np.array_equal(a,f+u)
  assert totals[pool,'scheduled_crop_exposure','Flip_selected'][3]==totals[pool,'scheduled_crop_exposure','Conf_selected'][3]
 table(ROOT/'SELECTOR_QUALITY.csv',out);table(ROOT/'SELECTOR_QUALITY_GAPS.csv',gaps);write(ROOT/'QUALITY_INPUT_SHA256.json',pins);write(ROOT/'QUALITY_ACCEPTANCE.json',dict(status='passed',all_endpoints_before_truth=True,partitions_exact=True,paired_exposure_requested_equal=True,L8_fit_gap_images=len(gaps),L8_no_new_registration=True,resources=resource_check()))
 print('Phase318 selectorquality: CMC/SPARCS audited; L8 fit reference gap explicit, val support complete')
if __name__=='__main__':main()
