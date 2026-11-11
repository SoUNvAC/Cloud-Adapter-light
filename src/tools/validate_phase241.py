"""Independent CPU arithmetic, identity, support and isolation acceptance."""
from phase241_cpu_common import *
import traceback

def independent_rank(s,y):
 # Vectorized ascending tie endpoints, separately implemented from the C++ descending merge.
 s=np.asarray(s,np.float64);y=np.asarray(y,np.int64);P=int(y.sum());N=len(y)-P
 if not P or not N:return None,None
 order=np.argsort(s);s=s[order];y=y[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];begins=np.r_[0,ends[:-1]+1];pos=np.add.reduceat(y,begins);neg=ends-begins+1-pos
 tp=np.cumsum(pos[::-1]);fp=np.cumsum(neg[::-1]);ap=float(np.sum(pos[::-1]*tp/(tp+fp))/P);below=np.cumsum(neg)-neg;auc=float(np.sum(pos*(below+.5*neg))/(P*N));return ap,auc

def main():
 global cpu
 cpu=CPU('validate');checks=[]
 def ok(name,value):assert bool(value),name;checks.append(name)
 lock=js(R/'ALIGNMENT_SEAL.json');ok('seal before test',lock['epoch']<js(R/'TEST_STARTED.json')['epoch'])
 for n,h in lock['files'].items():ok('sealed '+n,sha(R/n)==h)
 fitstart=js(R/'CALIBRATION_STARTED.json')['epoch'];ok('val completed before fit',js(R/'VAL_COMPLETE.json')['epoch']<=fitstart)
 positions=js(R/'CALIBRATION_POSITIONS.json');ok('six fits',len(js(R/'CALIBRATION_LOCK.json')['fits'])==6)
 samples={(v['domain'],v['scene']):v for v in positions['samples']};sourcegroups={d:grouped('val',d,'Source') for d in ('L8','SPARCS')};rng=np.random.default_rng(24101)
 for domain in ('L8','SPARCS'):
  for scene,rows in sourcegroups[domain].items():
   info=samples[domain,scene];chosen=np.sort(rng.choice(info['valid_population'],min(8192,info['valid_population']),replace=False));actual=[];offset=0;items={v['name']:v for v in info['selection']}
   for v in rows:
    with arrays(v) as (p,y):
     valid=np.flatnonzero(y.ravel()!=255)
     if v['name'] in items:
      item=items[v['name']];r=np.searchsorted(valid,np.asarray(item['indices']));ok('position valid '+v['name'],np.array_equal(valid[r],item['indices']));ok('position label '+v['name'],y.ravel()[item['indices']].tolist()==item['truth']);actual.extend((r+offset).tolist())
     offset+=len(valid)
   ok('same uniform positions '+scene,np.array_equal(chosen,actual));cpu.check()
 metrics=js(R/'METRICS.json');records=metrics['per_scene'];old={(v['domain'],v['split'],v['name'],v['model']):v for v in [json.loads(l) for l in (R/'EXPECTED236.jsonl').read_text().splitlines()]}
 for split in ('val','test'):
  index=cache_index(split);ok('exact index size '+split,len(index)==(2919 if split=='val' else 6531));keys=set();supports={};cm_by_scene={}
  for ordinal,v in enumerate(index):
   key=(v['domain'],split,v['name'],v['model']);ok('unique '+str(key),key not in keys);keys.add(key)
   ok('score hash '+str(key),sha(R/v['score_file'])==v['score_SHA']);ok('label hash '+str(key),sha(R/v['label_file'])==v['label_SHA'])
   with arrays(v) as (p,y):
    ok('float32 normalized '+str(key),p.dtype==np.float32 and p.shape==(3,512,512) and np.isfinite(p).all() and np.min(p)>=0 and np.max(abs(p.sum(0)-1))<1e-5)
    truth=y[y!=255];pr=np.argmax(p,axis=0).astype(np.uint8);mat=np.zeros((3,3),np.int64)
    for i in range(3):
     for j in range(3):mat[i,j]=np.count_nonzero((y==i)&(pr==j))
    ok('historical full SHA and counts '+str(key),hashlib.sha256(pr.tobytes()).hexdigest()==old[key]['prediction_SHA'] and mat.tolist()==old[key]['confusion']==v['confusion'])
    sk=(v['domain'],v['name']);support=tuple(int((y==i).sum()) for i in range(3))
    if sk in supports:ok('common support '+str(key),supports[sk]==support)
    supports[sk]=support;ck=(v['domain'],v['model'],v['scene']);cm_by_scene.setdefault(ck,np.zeros((3,3),np.int64));cm_by_scene[ck]+=mat
    if ordinal in (0,1,2,len(index)-3,len(index)-2,len(index)-1):
     identity=np.log(np.maximum(np.asarray(p,np.float64),1e-12));identity-=identity.max(0);identity=np.exp(identity);identity/=identity.sum(0);valid=y!=255
     checks.append('identity maxabs='+str(float(np.max(abs(identity-p)))))
     # Actual source scores may contain floor-induced ties; record numerical argmax differences.
     checks.append('identity argmax differences='+str(int(np.count_nonzero(identity.argmax(0)!=pr))))
     s=np.asarray(p[2][valid],np.float64);t=(truth==2).astype(np.uint8);a,b=independent_rank(s,t)
     from phase241_ranking import sorted_run,merge,clean
     folder=R/'scratch'/('validator_'+split+'_'+str(ordinal));folder.mkdir(parents=True,exist_ok=False);vtest=merge([sorted_run(folder,'x',s,t)],folder,'x');ok('independent actual tie AP/AUC '+str(key),(a is None and vtest['AP'] is None) or (abs(a-vtest['AP'])<1e-12 and abs(b-vtest['AUC'])<1e-12));clean(folder)
   if ordinal%100==0:cpu.check()
  for r in records:
   if r['split']==split and r['state']=='raw' and r['decision']=='argmax':ok('scene count reaggregation '+str(r['scene']),cm_by_scene[r['domain'],r['model'],r['scene']].tolist()==r['confusion'])
 # Independent pooled decisions and 2x2 algebra.
 for p in metrics['pooled']:
  rows=[r for r in records if all(r[k]==p[k] for k in ('domain','split','model','state','decision')) and (r['main'] if p['support']=='main' else (not r['main'] if p['support']=='nominal_no10' else True))]
  c=np.sum([r['confusion'] for r in rows],axis=0);ok('pooled common support',c.tolist()==p['confusion'])
  diag=np.diag(c);den=c.sum(0)+c.sum(1)-diag;value=np.mean(diag[den>0]/den[den>0]);ok('independent pooled mIoU',abs(value-p['mIoU_union'])<1e-13)
  if p['support']=='main':ok('main pooled reference three classes',bool(np.all(c.sum(1)>0)))
 # Independent shared-index bootstrap recomputation, including both-seed mean differences.
 indices=js(R/'BOOTSTRAP_INDICES.json')['domains'];published=js(R/'BOOTSTRAP.json')
 for domain,ix in indices.items():
  idx=np.asarray(ix['indices']);scenes=ix['scenes'];distributions={}
  for model in ('Source','C21701','C21702'):
   for state in ('raw','calibrated'):
    for decision in ('argmax','frozen_threshold'):
     lookup={r['scene']:r for r in records if all(r[k]==v for k,v in [('split','test'),('domain',domain),('model',model),('state',state),('decision',decision)])}
     if not all(s in lookup for s in scenes):continue
     c=np.asarray([lookup[s]['confusion'] for s in scenes],np.float64)[idx].sum(1);ref=c.sum(2);pred=c.sum(1);diag=np.diagonal(c,axis1=1,axis2=2);union=ref+pred-diag
     distributions[model,state,decision]={'mIoU':np.mean(diag/union,axis=1),'Shadow_recall':diag[:,2]/ref[:,2],'FPR':(c[:,0,2]+c[:,1,2])/(ref[:,0]+ref[:,1])}
    lookup={r['scene']:r['AUC'] for r in metrics['ranking'] if all(r[k]==v for k,v in [('split','test'),('domain',domain),('model',model),('state',state),('scope','scene'),('task','Shadow_Surface')])}
    if all(s in lookup for s in scenes):
     a=np.array([np.nan if lookup[s] is None else lookup[s] for s in scenes])[idx];n=np.isfinite(a).sum(1);distributions[model,state,None]={'Shadow_Surface_AUC':np.divide(np.nansum(a,1),n,out=np.full(10000,np.nan),where=n>0)}
  def delta(seed,metric,decision,contrast):
   A=distributions['Source','raw',decision][metric];B=distributions['Source','calibrated',decision][metric];C=distributions['C'+str(seed),'raw',decision][metric];D=distributions['C'+str(seed),'calibrated',decision][metric]
   return {'B-A':B-A,'D-C':D-C,'C-A':C-A,'D-B':D-B,'(B-A)-(D-C)':B-A-D+C}[contrast]
  for v in published:
   if v['domain']!=domain:continue
   a=(delta(21701,v['metric'],v['decision'],v['contrast'])+delta(21702,v['metric'],v['decision'],v['contrast']))/2 if v['seed']=='two_seed_difference_mean' else delta(v['seed'],v['metric'],v['decision'],v['contrast']);good=np.isfinite(a);ci=np.quantile(a[good],[.025,.975]) if good.any() else None
   ok('independent bootstrap '+str((domain,v['metric'],v['contrast'],v['seed'],v['decision'])),int((~good).sum())==v['NA_replicates'] and (ci is None if v['CI95'] is None else np.max(abs(ci-v['CI95']))<1e-12))
 calls=ledger();ok('call budget',len(calls)<=9600 and [v['ordinal'] for v in calls]==list(range(1,len(calls)+1)));counts=collections.Counter(v['kind'] for v in calls);ok('9450 full forwards',counts['full_input']==9450);ok('six repeats',counts['repeat_first_val']==6);ok('GPU budget',prior_seconds()<=14400)
 ok('kernel tests',js(R/'KERNEL_SELFTEST.json')['passed']);ok('no training',all('train' not in v['kind'] for v in calls));save('INDEPENDENT_ACCEPTANCE.json',{'passed':True,'checks':len(checks),'details':checks,'calls':dict(counts),'cumulative_calls':len(calls),'GPU_stage_seconds':prior_seconds(),'CPU_seconds_before_validation_finish':cpu_seconds()});cpu.finish()
if __name__=='__main__':
 try:main()
 except BaseException:
  save('VALIDATION_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'cpu' in globals():cpu.finish('failed')
  raise
