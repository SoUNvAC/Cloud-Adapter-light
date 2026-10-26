"""Metadata/cache SHA intake only. Never opens reference arrays or old target weights."""
from datetime import datetime,timezone,timedelta
from phase318_common import *
def main():
 assert sys.platform=='linux' and not ROOT.exists();assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261027'
 assert load(PRIOR/'STATUS.json')['status']=='completed' and load(PRIOR/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
 ROOT.mkdir()
 for n in ['tmp','cache','code']:(ROOT/n).mkdir()
 resource_check();pins={}
 def pin(p):
  p=Path(p)
  if str(p) not in pins:pins[str(p)]=sha(p)
  return p
 assert sha(pin(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'))==SOURCE
 for n,h in load(PRIOR/'GPU_CODE_LOCK.json').items():assert sha(pin(R/n))==h,n
 for n,h in load(pin(PRIOR/'DEPENDENCY_SHA256.json')).items():assert sha(pin(R/n))==h,n
 for n in ['RESULT.md','FROZEN_CONFIG.json','EXECUTION_LOCK.json','GPU_CODE_LOCK.json','DECISION_WINDOW.json','REFERENCE_LOCK.json','CPU_ACCEPTANCE.json','METRICS.csv','CPU_READOUT.json','PER_SCENE_METRICS.csv','PER_SCENE_NATIVE.csv']:pin(PRIOR/n)
 shutil.copyfile(PRIOR/'DECISION_WINDOW.json',ROOT/'DECISION_WINDOW.json')
 for pool in ['CMC','SPARCS','L8']:
  (ROOT/pool).mkdir()
  for n in ['IMAGES.json','SCHEDULE.json']:shutil.copyfile(pin(PRIOR/pool/n),ROOT/pool/n)
  data=inputs(pool)
  if pool=='L8':assert len(data['fit'])==100 and len(data['val'])==963 and all(r['usgs_shadows']=='yes' for r in data['fit']+data['val'])
  if pool=='CMC':
   q=load(pin(OLD/'PSEUDO_LOCK.json'));pin(OLD/'CROP_REPRODUCTION.csv');pin(OLD/'ENDPOINT_LOCK.json')
  else:q=load(pin(PRIOR/pool/'PSEUDO_LOCK.json'));pin(PRIOR/pool/'CROP_IDENTITY.csv')
  for e in q['rows']:
   path=OLD/'teacher'/(e['product']+'.npz') if pool=='CMC' else Path(e['file']);assert sha(pin(path))==e['sha256']
  for row in data['fit']+data['val']:
   if pool!='CMC':assert sha(pin(row['image_path']))==row['image_sha256']
  if pool=='CMC':
   ev=load(pin(OLD/'EVALUATION_LOCK.json'))
   for row in data['val']:
    for arm in ['Source','LoRA']:
     path=Path('/home/scv/shared/phase302_20261020/evaluation')/('Source_'+row['product']+'.npz') if arm=='Source' else OLD/'evaluation'/('LoRA_'+row['product']+'.npz')
     expected=next(r['sha256'] for r in ev['rows'] if r['product']==row['product'] and r['arm']==arm);assert sha(pin(path))==expected
  for arm in ['CB'] if pool=='CMC' else ['Source','PL','CB']:
   e=load(pin(PRIOR/pool/arm/'EVALUATION_LOCK.json'))
   for r in e['rows']:assert sha(pin(r['file']))==r['sha256']
 # Conservative uncompressed bounds: qflip/pred/valid; two student evaluation sets;
 # probes and code/curves. No copies of historical arrays, labels or original scores.
 flip=(100*1022**2+160*512**2)*6+260*8192
 evaluation=2*(32*1022**2+973*512**2)*6+2010*8192
 probes=40*512**2*16
 raw_upper=flip+evaluation+probes+150000000
 projected=2*raw_upper+1000000000;assert projected<=12000000000
 calls=1060+5812+2522+40+200;assert calls<=11000
 paths=list((R/'src/tools').glob('*phase318*.py'))
 code={p.relative_to(R).as_posix():sha(p) for p in paths}
 for p in paths:shutil.copyfile(p,ROOT/'code'/p.name)
 write(ROOT/'GPU_CODE_LOCK.json',code)
 write(ROOT/'EXECUTION_LOCK.json',dict(inputs=pins,code=code,source_sha256=SOURCE,seed=30201,arms=ARMS,max_forwards=11000,forward_upper=calls,max_gpu_seconds=43200,max_CPU_seconds=7200,max_twohosts_bytes=12000000000,estimated_twohosts_bytes=projected,estimated_uncompressed_new=raw_upper,original_references_read=False,resources=resource_check()))
 write(ROOT/'FROZEN_CONFIG.json',dict(strategy='selective CB normalized by sum weights; Surface/Cloud CB, selected Shadow CB, unselected Shadow PL; no clamping',threshold=.9,flip='horizontal per original CMC window then inverse and score stitch; otherpools flip preprocessed-input precursor',conf='per-step top original qShadow, originalfullimage rowmajor ties, same k as Flip',no_reference_until_all_endpoints_and_evaluation_sealed=True,fit_L8_audit='no new registration: if fit rawfill coordinates not previously verified, declare uncovered',decision_window_sha=sha(ROOT/'DECISION_WINDOW.json'),execution_sha=sha(ROOT/'EXECUTION_LOCK.json')))
 write(ROOT/'STAGES.json',{});write(ROOT/'STATUS.json',dict(status='prepared',forward_upper=calls,persistent_upper=projected));print(json.dumps(dict(forward_upper=calls,persistent_upper=projected)))
if __name__=='__main__':main()
