"""Seal controlled shared score delivery; no labels, model calls or work_dirs."""
import hashlib,json,resource,zipfile
from pathlib import Path
import numpy as np
R=Path('/home/scv/shared/phase203_20261018');STATES=('shared_R_00','12201_11','12301_11','12201_10','12201_01','12301_10','12301_01')
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 s=load(R/'RUN_STATUS.json');assert s['status']=='completed_pending_local_delivery' and s['calls']==4865 and s['elapsed_seconds']<=7200 and (R/'EXIT_CODE').read_text().strip()=='0';assert s['completed']==list(STATES)
 assert sum(load(R/t/'COMPLETE.json')['all_calls'] for t in STATES)+7==s['calls']
 files={};support=[]
 def add(p):assert p.is_file() and p.resolve().is_relative_to(R.resolve());files[p.relative_to(R).as_posix()]=sha(p)
 for n in ('RUN_STATUS.json','LAUNCH.json','EXECUTION_LOCK.json','RECOVERY_LOCK.json','WHITELIST.json','EVALUATION_INPUT_PLAN.json','OLD_SCORE_INDEX.json','CALL_LEDGER_RECONCILIATION_03.json','PREFLIGHT.json'):add(R/n)
 originals=load(R/'OLD_SCORE_INDEX.json')['rows'];old={(a['state'],a['pool'],a['product_id']):a for a in originals}
 for t in STATES:
  d=R/t;assert load(d/'COMPLETE.json')['state_unchanged_in_eval'];add(d/'COMPLETE.json');add(d/'STATE_IDENTITY.json');add(d/'CONTROLLER_STATUS.json')
  if t in STATES[:3]:
   for pool in ('Catalogue','PixBox'):assert load(d/(pool+'_REPRODUCTION.json'))['passed'];add(d/(pool+'_REPRODUCTION.json'))
  for pool in ('Catalogue','PixBox'):
   st=d/('R_'+pool);assert (st/'EXIT_CODE').read_text().strip()=='0'
   for n in ('SCORES_SEALED.json','SCENE_RECEIPTS.json','MODEL_INFO.json','RUN_STATUS.json'):add(st/n)
   seals=load(st/'SCORES_SEALED.json')['scores_sha256'];assert len(seals)==(1 if t in STATES[:3] else (32 if pool=='Catalogue' else 16))
   for key,h in seals.items():
    q,pid=key.split('/',1);p=st/'scores'/q/(pid+'.npz' if q=='Catalogue' else pid+'/UNLABELLED_POINT_SCORES.json');assert sha(p)==h;add(p)
    base=old['shared_R_00',pool,pid];bp=Path(base['path']);assert sha(bp)==base['sha256']
    if pool=='Catalogue':
     with np.load(p,allow_pickle=False) as a,np.load(bp,allow_pickle=False) as b:
      assert np.isfinite(a['scores']).all() and np.array_equal(a['flat_indices'],b['flat_indices']) and np.array_equal(a['valid'],b['valid']);valid=int(a['valid'].sum())
    else:
     aa={v['point_id']:v for v in load(p)['points']};bb={v['point_id']:v for v in load(bp)['points']};assert set(aa)==set(bb)
     for k,v in aa.items():assert v['valid']==bb[k]['valid'] and (v['shadow_confidence'] is None if not v['valid'] else np.isfinite(v['shadow_confidence']))
     valid=sum(v['valid'] for v in aa.values())
    support.append(dict(state=t,pool=pool,product_id=pid,valid=valid,unchanged_vs_R=True))
 for p in (R/'code').glob('*.py'):add(p)
 for archive in R.glob('engineering_failure_*'):
  for p in archive.rglob('*'):
   if p.is_file() and p.suffix in ('.json','.log','.py') and '/scores/' not in p.as_posix():add(p)
 manifest=dict(files=files,support=support,score_files=198,neural_calls=s['calls'],active_stage_wall_seconds=s['elapsed_seconds'],CPU_only_collection=True,reference_labels_accessed=False,work_dirs_transferred=False)
 archive=R/'phase203_delivery.zip';assert not archive.exists()
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  for name in sorted(files):z.write(R/name,name)
  z.writestr('DELIVERY_MANIFEST.json',json.dumps(manifest,indent=2))
 total=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());assert total<6_000_000_000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3
 receipt=dict(status='sealed',sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),neural_calls=4865,active_stage_wall_seconds=s['elapsed_seconds'],score_files=len(support),reference_labels_uploaded=0,new_neural_calls=0,shared_total_logical_bytes=total)
 with (R/'DELIVERY_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))
if __name__=='__main__':main()
