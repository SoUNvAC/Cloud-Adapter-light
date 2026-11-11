"""Independent integer/loop/float64 CPU validation, including the complete fixed bootstrap."""
from pathlib import Path
import csv,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase221/phase221_20261024';RAW=OUT/'delivery';S=[21701,21702];A=[0,.25,.5,.75,1]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def score(c):
 v=[]
 for k in range(3):
  u=c[k].sum()+c[:,k].sum()-c[k,k]
  if u:v.append(c[k,k]/u)
 return float(sum(v)/len(v)*100)
def batch(c):
 v=[]
 for k in range(3):
  u=c[:,k,:].sum(1)+c[:,:,k].sum(1)-c[:,k,k];v.append(np.divide(c[:,k,k],u,out=np.full(len(c),np.nan),where=u>0))
 return np.nanmean(np.stack(v,1),1)*100
def main():
 clock=time.monotonic();manifest=js(RAW/'DELIVERY_MANIFEST.json');assert all((RAW/n).stat().st_size==v['bytes'] and sha(RAW/n)==v['SHA'] for n,v in manifest.items());remote=js(RAW/'REMOTE_CPU_ACCEPTANCE.json');assert remote['status']=='passed' and remote['total_units']==75018 and len(remote['independent_class_normalization_weight_proofs'])==256
 # Independently decode every delivered mask; confidence/permutation reconstruction was CPU-checked remotely.
 for r in js(RAW/'weight_preparation/WEIGHT_MANIFEST.json'):
  sid=r['sample_id'];path=RAW/f'weight_preparation/masks/{sid}.npz';assert sha(path)==r['packed_masks_SHA']
  with np.load(path,allow_pickle=False) as z:
   teacher=z['teacher_prediction'].reshape(-1);g=np.unpackbits(z['G_bits'],count=262144).astype(np.float32)+.5;ph=np.unpackbits(z['P_bits'],count=262144);n=np.empty_like(g);p=np.empty_like(g)
   assert hashlib.sha256(g.tobytes()).hexdigest()==r['G_SHA'] and hashlib.sha256(teacher.tobytes()).hexdigest()==r['teacher_prediction_SHA']
   for c in range(3):
    ids=np.flatnonzero(teacher==c);q=r['classes'][c];assert len(ids)==q['support']
    if not len(ids):assert q['mean_g'] is None;continue
    mean=float(g[ids].astype(np.float64).mean());assert mean==q['mean_g'];n[ids]=(g[ids].astype(np.float64)/mean).astype(np.float32);p[ids]=((.5+ph[ids].astype(np.float64))/mean).astype(np.float32);assert abs(n[ids].mean(dtype=np.float64)-1)<=1e-6 and abs(p[ids].mean(dtype=np.float64)-1)<=1e-6 and np.array_equal(np.sort(n[ids]),np.sort(p[ids]));assert q['seed']==int.from_bytes(hashlib.sha256(('phase221|'+sid+'|'+str(c)).encode()).digest()[:8],'big')
   assert hashlib.sha256(n.tobytes()).hexdigest()==r['N_SHA'] and hashlib.sha256(p.tobytes()).hexdigest()==r['P_SHA']
 cases={};pool={(int(r['seed']),r['arm'],float(r['alpha']),r['pool']):r for r in rows('ALL_POOL_METRICS.csv')}
 for r in rows('PER_GROUP.csv'):
  key=int(r['seed']),r['arm'],float(r['alpha']),r['pool'];c=np.array(json.loads(r['confusion']),np.int64);assert c.shape==(3,3) and (c>=0).all();cases.setdefault(key,{})[r['group']]=c;assert abs(score(c)-float(r['mIoU']))<1e-10
  for i,n in enumerate(['Surface','Cloud','Shadow']):
   u=c[i].sum()+c[:,i].sum()-c[i,i]
   if u:assert abs(float(r[n+'_IoU'])-100*c[i,i]/u)<1e-10
  if c[2].sum():assert abs(float(r['Shadow_recall'])-c[2,2]/c[2].sum())<1e-12
  if c[:2].sum():assert abs(float(r['nonShadow_to_Shadow_FPR'])-c[:2,2].sum()/c[:2].sum())<1e-12
 assert len(cases)==180
 for key,g in cases.items():
  b=cases[21701,'C',0,key[3]];assert set(g)==set(b) and all(np.array_equal(v.sum(1),b[k].sum(1)) for k,v in g.items());assert abs(score(sum(g.values()))-float(pool[key]['mIoU']))<1e-10
 official=list(csv.DictReader((ROOT/'outputs/phase65/source_metadata/acquired/val_metadata.csv').open(encoding='utf-8-sig')));support=js(RAW/'evaluation/INPUT_SUPPORT_LOCK.json')['source'];assert len(support)==535 and all(v['scene']==official[int(Path(v['name']).stem)]['roi_id'] for v in support)
 base=cases[21701,'C',0,'source'];stats={};chosen={}
 for s in S:
  for a in ['C','D','G','S','N','P']:
   for f in A:
    g=cases[s,a,f,'source'];loss=sorted([score(base[k])-score(g[k]) for k in base],reverse=True);stats[s,a,f]={'pool':score(sum(base.values()))-score(sum(g.values())),'tail':sum(loss[:11])/11,'worst':loss[0],'over5':sum(v>5 for v in loss)}
   for b in [1,3,5]:chosen[s,a,b]=max(f for f in A if stats[s,a,f]['pool']<=b and stats[s,a,f]['tail']<=5)
 for r in rows('ALL_CANDIDATES.csv'):
  st=stats[int(r['seed']),r['arm'],float(r['alpha'])]
  for name,key in [('pool_loss_pp','pool'),('T11_pp','tail'),('worst_ROI_loss_pp','worst'),('ROI_loss_gt5','over5')]:assert abs(float(r[name])-st[key])<1e-10
 for r in rows('JOINT_RULE.csv'):assert float(r['alpha'])==chosen[int(r['seed']),r['arm'],int(r['budget_pp'])]
 # Teacher/reference/candidate counts reconstruct all states and both distinct net components.
 tensors={}
 for r in rows('ENDPOINT_TEACHER_CORRECTNESS.csv'):
  s=int(r['seed']);a=r['arm'];k=r['ROI'];t=np.array(json.loads(r['reference_teacher_student_counts']),np.int64);assert t.shape==(3,3,3) and (t>=0).all() and np.array_equal(t.sum(1),cases[s,a,1,'source'][k]) and np.array_equal(t.sum(2),base[k]);tensors[s,a,k]=t
  for tc in [0,1]:
   for sc in [0,1]:assert int(r[f'T{tc}S{sc}'])==sum(int(t[i,j,z]) for i in range(3) for j in range(3) for z in range(3) if int(i==j)==tc and int(i==z)==sc)
 def counts(t):
  protected=sum(int(t[i,i,z]) for i in range(3) for z in range(3) if z!=i);corrected=sum(int(t[i,j,i]) for i in range(3) for j in range(3) if j!=i);err=int(t.sum()-sum(t[i,j,i] for i in range(3) for j in range(3)));return protected,corrected,err
 for r in rows('ALL_ROI_CORRECTNESS_DIFFERENCES.csv'):
  s=int(r['seed']);a,b=r['comparison'].split('-');k=r['ROI'];cp,cc,ce=counts(tensors[s,a,k]);bp,bc,be=counts(tensors[s,b,k]);benefit=bp-cp;cost=bc-cc;assert ce-be==cost-benefit
  assert int(r['teacher_correct_net_benefit_pixels'])==benefit and int(r['teacher_correction_net_cost_pixels'])==cost and int(r['total_error_delta_pixels'])==ce-be
 for r in rows('TABLE_B_DIFFERENCES.csv'):
  if r['seed']=='mean_of_two':continue
  rr=[v for v in rows('ALL_ROI_CORRECTNESS_DIFFERENCES.csv') if v['seed']==r['seed'] and v['comparison']==r['comparison']];assert len(rr)==107
  for k in ['teacher_correct_net_benefit_pixels','teacher_correction_net_cost_pixels','total_error_delta_pixels']:assert sum(int(v[k]) for v in rr)==int(r[k])
 bands=rows('TEACHER_CONFIDENCE_DIAGNOSTIC.csv')
 for r in bands:
  cm=np.array(json.loads(r['confusion']),np.int64);assert int(r['valid_pixels'])==cm.sum() and int(r['teacher_errors'])==cm.sum()-np.trace(cm)
  if cm.sum():assert abs(float(r['reference_error_rate'])-(cm.sum()-np.trace(cm))/cm.sum())<1e-12
  for i,n in enumerate(['Surface','Cloud','Shadow']):assert int(r[n+'_support'])==cm[i].sum() and int(r[n+'_errors'])==cm[i].sum()-cm[i,i]
 for k in list(base)+['__pooled__']:
  cm=sum(np.array(json.loads(r['confusion']),np.int64) for r in bands if r['ROI']==k);assert np.array_equal(cm,sum(base.values()) if k=='__pooled__' else base[k])
 ci={(r['comparison'],r['context'],r['scope'],r['metric']):r for r in rows('PAIRED_CI.csv')};proof=js(OUT/'BOOTSTRAP_PROOF.json');checks=0;scalar=0
 for a,b in [('N','G'),('N','D'),('N','P')]:
  for ctx in ['endpoint_alpha1','B1','B3','B5']:
   values={}
   for s in S:
    af,bf=(1,1) if ctx=='endpoint_alpha1' else (chosen[s,a,int(ctx[1:])],chosen[s,b,int(ctx[1:])]);values[str(s)]={}
    for p,n in [('test',10),('source',107)]:
     ca=cases[s,a,af,p];cb=cases[s,b,bf,p];ids=sorted(ca);assert len(ids)==n;ix=np.random.default_rng(217023).integers(0,n,(10000,n));aa=np.stack([ca[k] for k in ids]);bb=np.stack([cb[k] for k in ids]);av=batch(aa[ix].sum(1));bv=batch(bb[ix].sum(1))
     if p=='test':values[str(s)]['test_mIoU_diff_pp']=av-bv
     else:
      values[str(s)]['source_pool_loss_diff_pp']=bv-av;al=np.array([score(base[k])-score(ca[k]) for k in ids]);bl=np.array([score(base[k])-score(cb[k]) for k in ids]);values[str(s)]['T11_diff_pp']=np.sort(al[ix],axis=1)[:,-11:].mean(1)-np.sort(bl[ix],axis=1)[:,-11:].mean(1)
     for j,index in enumerate(ix[:20]):
      names=[ids[i] for i in index]
      if p=='test':vv={'test_mIoU_diff_pp':score(sum(ca[k] for k in names))-score(sum(cb[k] for k in names))}
      else:
       al=sorted([score(base[k])-score(ca[k]) for k in names],reverse=True);bl=sorted([score(base[k])-score(cb[k]) for k in names],reverse=True);vv={'source_pool_loss_diff_pp':score(sum(cb[k] for k in names))-score(sum(ca[k] for k in names)),'T11_diff_pp':sum(al[:11])/11-sum(bl[:11])/11}
      for metric,v in vv.items():assert abs(v-values[str(s)][metric][j])<1e-10;scalar+=1
   values['mean_of_two']={k:(values[str(S[0])][k]+values[str(S[1])][k])/2 for k in values[str(S[0])]}
   for scope,ds in values.items():
    for k,v in ds.items():
     row=ci[a+'-'+b,ctx,scope,k];lo,hi=np.quantile(v,[.025,.975]);assert abs(lo-float(row['CI95_low']))<1e-10 and abs(hi-float(row['CI95_high']))<1e-10 and np.allclose(v[:50],proof[a+'-'+b+'/'+ctx+'/'+scope+'/'+k]['first50'],rtol=0,atol=1e-10);checks+=1
 # Verify the point estimates independently, including selected alpha pairs.
 for r in rows('ALL_COMPARISONS.csv'):
  s=int(r['seed']);a,b=r['comparison'].split('-');af=float(r['candidate_alpha']);bf=float(r['baseline_alpha']);expect={'test_mIoU_diff_pp':score(sum(cases[s,a,af,'test'].values()))-score(sum(cases[s,b,bf,'test'].values())),'source_pool_loss_diff_pp':stats[s,a,af]['pool']-stats[s,b,bf]['pool'],'T11_diff_pp':stats[s,a,af]['tail']-stats[s,b,bf]['tail']}
  for k,v in expect.items():assert abs(v-float(r[k]))<1e-10 and abs(v-float(ci[r['comparison'],r['context'],str(s),k]['point']))<1e-10
 for r in rows('TABLE_B_DIFFERENCES.csv'):
  if r['seed']=='mean_of_two':
   rr=[q for q in rows('TABLE_B_DIFFERENCES.csv') if q['seed']!='mean_of_two' and q['comparison']==r['comparison']]
   for k in ['test_mIoU_diff_pp','source_pool_loss_diff_pp','T11_diff_pp','Shadow_error_delta_pixels']:assert abs(float(r[k])-np.mean([float(v[k]) for v in rr]))<1e-10
  else:
   s=int(r['seed']);a,b=r['comparison'].split('-');aa=sum(cases[s,a,1,'source'].values());bb=sum(cases[s,b,1,'source'].values());assert int(r['Shadow_error_delta_pixels'])==int(aa[2].sum()-aa[2,2]-bb[2].sum()+bb[2,2])
 result={'status':'passed','seconds':time.monotonic()-clock,'raw_artifacts_SHA_checked':len(manifest),'all256_local_mask_histograms_and_SHA':True,'remote_independent_class_normalization_PCG64_numpy1264':True,'full10000_CI_checks':checks,'scalar_bootstrap_checks':scalar,'same107_ROIs_10_test_scenes':True,'no_seed_group_inflation':True,'all_correctness_identities':True,'confidence_supports_partition_Source':True,'source_rules_independently_recomputed':True,'new_model_calls':0,'code_SHA':sha(Path(__file__))};(OUT/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
