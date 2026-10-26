"""Image-only manifest/schedule/cache preparation. No target reference pixels."""
from collections import defaultdict,Counter
from datetime import datetime,timezone,timedelta
import subprocess
import numpy as np
from phase317_common import *
def main():
 assert sys.platform=='linux' and not ROOT.exists();assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261026'
 ROOT.mkdir();(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir();(ROOT/'code').mkdir();resource_check();pins={}
 def pin(p):p=Path(p);pins[str(p)]=sha(p);return p
 source=R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(pin(source))==SOURCE
 for p in (R/'src/tools').glob('*phase317*.py'):shutil.copyfile(p,ROOT/'code'/p.name)
 for name in ['worker_phase304.py','execute_phase302.py','phase302_images.py']:
  p=R/'src/tools'/name;assert sha(p)==load(OLD/'EXECUTION_LOCK.json')['code']['src/tools/'+name];pin(p)
 for name in ['PSEUDO_LOCK.json','CROP_SCHEDULE.json','CROP_REPRODUCTION.csv','FIT_IMAGES.json','EVAL_IMAGES.json','ENDPOINT_LOCK.json','EVALUATION_LOCK.json']:
  pin(OLD/name)
 assert load(OLD/'PSEUDO_LOCK.json')['expected_updates']==955
 (ROOT/'CMC').mkdir();write(ROOT/'CMC/IMAGES.json',dict(fit=load(OLD/'FIT_IMAGES.json')['rows'],val=load(OLD/'EVAL_IMAGES.json')['rows'],input='original1022TOA RGB*255 thenBGR; originalnine512windows'))
 shutil.copyfile(OLD/'CROP_SCHEDULE.json',ROOT/'CMC/SCHEDULE.json')
 for r in load(OLD/'PSEUDO_LOCK.json')['rows']:
  p=OLD/'teacher'/(r['product']+'.npz');assert sha(pin(p))==r['sha256']
 sm=pin(R/'src/work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv');lm=pin(R/'src/work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv');ss=pin(R/'src/research_plans/protocol_data/l8_biome_usgs_shadow_status.csv')
 statuses={r['scene']:r['usgs_shadows'] for r in read(ss)};assert len(statuses)==96 and sum(v=='yes' for v in statuses.values())==32
 def resolve(p):
  p=Path(p);p=p if p.is_absolute() else R/'src'/p;p=p.resolve();assert p.is_relative_to(R) or p.is_relative_to(Path('/home/scv/shared'));assert p.is_file(),p;return p
 counts=[]
 for pool,manifest in [('SPARCS',sm),('L8',lm)]:
  d=ROOT/pool;d.mkdir();rows=read(manifest);scenes={}
  for r in rows:
   assert scenes.setdefault(r['scene'],r['new_split'])==r['new_split'],'scene crosses pools'
  train=[r for r in rows if r['new_split']=='target_train' and (pool!='L8' or statuses[r['scene']]=='yes')];val=[r for r in rows if r['new_split']=='target_val' and (pool!='L8' or statuses[r['scene']]=='yes')]
  assert val and len(val)<=1200 and train
  for r in train+val:
   r['manifest_image_path']=r['image_path']
   r['image_path']=str(resolve(r['image_path']));r['mask_path']=str(resolve(r['mask_path']))
   if pool=='L8':r['usgs_shadows']=statuses[r['scene']]
  if pool=='L8':
   buckets=defaultdict(list)
   for r in train:
    rel=Path(r['manifest_image_path']);assert not rel.is_absolute(),'L8 manifestmember must retain its complete originalrelativepath'
    r['selection_relative_path']=rel.as_posix();r['selection_hash']=hashlib.sha256(('phase317-fit:'+rel.as_posix()).encode()).hexdigest();buckets[r['scene']].append(r)
   for v in buckets.values():v.sort(key=lambda r:(r['selection_hash'],r['name']))
   selected=[];level=0
   while len(selected)<100 and any(level<len(v) for v in buckets.values()):
    for sc in sorted(buckets):
     if level<len(buckets[sc]) and len(selected)<100:selected.append(buckets[sc][level])
    level+=1
   train=selected
  else:assert len(train)==60 and len(val)==10
  fit=sorted(train,key=lambda r:(r['scene'],r['name']));val.sort(key=lambda r:r['name'])
  assert not ({r['scene'] for r in fit}&{r['scene'] for r in val})
  for r in fit+val:r['image_sha256']=sha(pin(r['image_path']))
  write(d/'IMAGES.json',dict(fit=fit,val=val,input='existingdisplayRGB uint8; mmseg Resize512bilinear then SourceBGRtoRGB mean/std; all finitepixels imagevalid includingblack',selection='SHA256 phase317-fit: resolvedfullmemberrelativepath, roundrobin scene; no labels/predictions',fit_scene_counts=dict(Counter(r['scene'] for r in fit)),val_scene_counts=dict(Counter(r['scene'] for r in val))))
  rng=np.random.Generator(np.random.PCG64(31700));steps=[dict(attempt=i+1,fit_index=int(rng.integers(len(fit))),horizontal_flip=bool(rng.integers(2))) for i in range(1000)];write(d/'SCHEDULE.json',dict(seed=31700,generator='PCG64',rows=steps))
  counts.append(dict(pool=pool,fit=len(fit),val=len(val),fit_scenes=len({r['scene'] for r in fit}),val_scenes=len({r['scene'] for r in val})))
 # Conservative uncompressed output estimate: streaming backup, no persistent second archive.
 nval=sum(r['val'] for r in counts);nfit=sum(r['fit'] for r in counts)
 raw_upper=3*nval*(512*512*6+4096)+32*(1022*1022*6+4096)+nfit*(512*512*16+4096)+24*(512*512*16+4096)+100000000
 projected=2*raw_upper+1000000000;calls=955+288+16+4000+nfit+3*nval+120
 assert projected<=16000000000 and calls<=11000
 code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase317*.py')}
 write(ROOT/'EXECUTION_LOCK.json',dict(source_sha256=SOURCE,inputs=pins,code=code,seed=30201,rank=4,alpha=4,dropout=0,parameters=294912,threshold=.9,arms=ARMS,counts=counts,forward_upper_bound=calls,max_forwards=11000,max_gpu_seconds=43200,estimated_uncompressed_new_bytes=raw_upper,estimated_twohosts_including_summary=projected,backup='stream with no persistent large archive',training_valid='imageonly; finite displayblack retained; CMC originalvalid',source_and_history='historicallydeveloped pools, not blind; no oldtargetheads/optimizers',resources=resource_check()))
 table(ROOT/'DATA_COUNTS.csv',counts);write(ROOT/'STAGES.json',{});write(ROOT/'STATUS.json',dict(status='prepared',counts=counts,forward_upper=calls,persistent_upper=projected))
 print(json.dumps(dict(counts=counts,forward_upper=calls,persistent_upper=projected)))
if __name__=='__main__':main()
