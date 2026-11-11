"""Independent217 integer support, scalar means and shared-index bootstrap audit."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase217/phase217_20261023';RAW=OUT/'delivery';S=[21701,21702]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def score(c):
 c=np.asarray(c);rat=[]
 for k in range(3):
  u=c[k].sum()+c[:,k].sum()-c[k,k]
  if u:rat.append(c[k,k]/u)
 return float(sum(rat)/len(rat)*100)
def bscore(c):
 v=[]
 for k in range(3):
  u=c[:,k,:].sum(1)+c[:,:,k].sum(1)-c[:,k,k];v.append(np.divide(c[:,k,k],u,out=np.full(len(c),np.nan),where=u>0))
 return np.nanmean(np.stack(v,1),1)*100
def main():
 manifest=js(RAW/'DELIVERY_MANIFEST.json');assert all((RAW/n).stat().st_size==r['bytes'] and hashlib.sha256((RAW/n).read_bytes()).hexdigest()==r['SHA'] for n,r in manifest.items())
 assert js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed'
 cases={}
 for r in rows('PER_GROUP.csv'):
  key=int(r['seed']),r['arm'],float(r['alpha']),r['pool'];c=np.asarray(json.loads(r['confusion']),np.int64);assert c.shape==(3,3) and (c>=0).all();cases.setdefault(key,{})[r['group']]=c;assert abs(score(c)-float(r['mIoU']))<1e-10
  for k,label in enumerate(['Surface','Cloud','Shadow']):
   den=c[k].sum()+c[:,k].sum()-c[k,k]
   if den:assert abs(float(r[label+'_IoU'])-100*c[k,k]/den)<1e-10
  if c[2].sum():assert abs(float(r['Shadow_recall'])-c[2,2]/c[2].sum())<1e-12
  if c[:2].sum():assert abs(float(r['nonShadow_to_Shadow_FPR'])-c[:2,2].sum()/c[:2].sum())<1e-12
 pools={(int(r['seed']),r['arm'],float(r['alpha']),r['pool']):r for r in rows('ALL_POOL_METRICS.csv')}
 for key,gs in cases.items():
  reference=cases[S[0],'C',0,key[3]];assert set(gs)==set(reference) and all(np.array_equal(c.sum(1),reference[g].sum(1)) for g,c in gs.items())
  assert abs(score(sum(gs.values()))-float(pools[key]['mIoU']))<1e-10
 base=cases[S[0],'C',0,'source'];assert len(base)==107
 official=list(csv.DictReader((ROOT/'outputs/phase65/source_metadata/acquired/val_metadata.csv').open(encoding='utf-8-sig')));support=js(RAW/'evaluation/INPUT_SUPPORT_LOCK.json')['source'];assert all(r['scene']==official[int(Path(r['name']).stem)]['roi_id'] for r in support)
 chosen={};stats={}
 for s in S:
  for a in ['C','D']:
   for f in [0,.25,.5,.75,1]:
    gs=cases[s,a,f,'source'];assert set(gs)==set(base) and all(np.array_equal(c.sum(1),base[g].sum(1)) for g,c in gs.items());loss=sorted([score(base[g])-score(gs[g]) for g in gs],reverse=True);stats[s,a,f]={'pool':score(sum(base.values()))-score(sum(gs.values())),'tail':sum(loss[:11])/11}
   for b in [1,3,5]:chosen[s,a,b]=max(f for f in [0,.25,.5,.75,1] if stats[s,a,f]['pool']<=b and stats[s,a,f]['tail']<=5)
 lock=js(RAW/'evaluation/SOURCE_SELECTION_LOCK.json');assert not lock['target_metrics_opened_before_lock'];assert all(r['alpha']==chosen[r['seed'],r['arm'],r['budget_pp']] for r in lock['rows'])
 for r in rows('ALL_CANDIDATES.csv'):
  t=stats[int(r['seed']),r['arm'],float(r['alpha'])];assert abs(t['pool']-float(r['pool_loss_pp']))<1e-10 and abs(t['tail']-float(r['T11_pp']))<1e-10
 # Full10000 draws, independently recompute nonlinear metrics inside each trained seed.
 ci={(r['context'],r['scope'],r['metric']):r for r in rows('PAIRED_CI.csv')};proof=js(OUT/'BOOTSTRAP_PROOF.json');checks=0;scalar_checks=0
 for context in ['endpoint_alpha1','B1','B3','B5']:
  values={}
  for s in S:
   ca,da=(1,1) if context=='endpoint_alpha1' else (chosen[s,'C',int(context[1:])],chosen[s,'D',int(context[1:])]);values[str(s)]={}
   for pool,n in [('test',10),('source',107)]:
    ids=sorted(cases[s,'C',ca,pool]);assert len(ids)==n;ix=np.random.default_rng(217023).integers(0,n,(10000,n));c=cases[s,'C',ca,pool];d=cases[s,'D',da,pool];cc=np.stack([c[g] for g in ids]);dd=np.stack([d[g] for g in ids]);cv=bscore(cc[ix].sum(1));dv=bscore(dd[ix].sum(1))
    if pool=='test':values[str(s)]['test_mIoU_diff_pp']=dv-cv
    else:
     values[str(s)]['source_pool_loss_diff_pp']=cv-dv;cl=np.array([score(base[g])-score(c[g]) for g in ids]);dl=np.array([score(base[g])-score(d[g]) for g in ids]);values[str(s)]['T11_diff_pp']=np.sort(dl[ix],axis=1)[:,-11:].mean(1)-np.sort(cl[ix],axis=1)[:,-11:].mean(1)
    for j,indices in enumerate(ix[:20]):
     names=[ids[i] for i in indices]
     if pool=='test':v={'test_mIoU_diff_pp':score(sum(d[g] for g in names))-score(sum(c[g] for g in names))}
     else:
      cll=sorted([score(base[g])-score(c[g]) for g in names],reverse=True);dll=sorted([score(base[g])-score(d[g]) for g in names],reverse=True);v={'source_pool_loss_diff_pp':score(sum(c[g] for g in names))-score(sum(d[g] for g in names)),'T11_diff_pp':sum(dll[:11])/11-sum(cll[:11])/11}
     for k,x in v.items():assert abs(values[str(s)][k][j]-x)<1e-10;scalar_checks+=1
  values['mean_of_two']={k:np.add(values[str(S[0])][k],values[str(S[1])][k])/2 for k in values[str(S[0])]}
  for scope,vals in values.items():
   for k,x in vals.items():
    r=ci[context,scope,k];lo,hi=np.quantile(x,[.025,.975]);assert abs(lo-float(r['CI95_low']))<1e-10 and abs(hi-float(r['CI95_high']))<1e-10
    assert np.allclose(x[:50],proof[context+'/'+scope+'/'+k]['first50'],rtol=0,atol=1e-10);checks+=1
 tables=rows('ENDPOINTS.csv')+rows('JOINT_RULE.csv')
 for r in tables:
  if r['seed']=='mean_of_21701_21702':
   same=[x for x in tables if x['context']==r['context'] and x['seed']!='mean_of_21701_21702'];assert len(same)==2
   for k in r:
    if k not in ['seed','context','C_alpha','D_alpha']:assert abs(float(r[k])-np.mean([float(x[k]) for x in same]))<1e-10
  else:
   s=int(r['seed']);ca=float(r['C_alpha']);da=float(r['D_alpha'])
   for arm,alpha in [('C',ca),('D',da)]:
    assert abs(float(r[arm+'_test_mIoU'])-score(sum(cases[s,arm,alpha,'test'].values())))<1e-10
    assert abs(float(r[arm+'_pool_loss_pp'])-stats[s,arm,alpha]['pool'])<1e-10 and abs(float(r[arm+'_T11_pp'])-stats[s,arm,alpha]['tail'])<1e-10
   for stem,name in [('test_mIoU','test_mIoU_diff_pp'),('pool_loss_pp','source_pool_loss_diff_pp'),('T11_pp','T11_diff_pp')]:
    diff=float(r['D_'+stem])-float(r['C_'+stem]);key='D_minus_C_'+('test_mIoU_pp' if stem=='test_mIoU' else stem);assert abs(float(r[key])-diff)<1e-10 and abs(float(ci[r['context'],str(s),name]['point'])-diff)<1e-10
 result={'status':'passed','same_official107_ROIs_and10_test_scenes':True,'no_seed_group_inflation':True,'source_rules_independently_recomputed':True,'full10000_CI_checks':checks,'scalar_bootstrap_checks':scalar_checks,'means_of_within_seed_metrics_not_confusion':True,'all8000_training_parent_masks_and_all_heads_remote_independent_check':True,'new_model_calls':0,'code_SHA':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
