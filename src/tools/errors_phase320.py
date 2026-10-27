"""Full fixed fit-pool native truth, boundary and original geometry decomposition."""
from phase320_common import *
from collections import defaultdict
import numpy as np
from scipy.ndimage import binary_dilation
NATIVE={'SPARCS':[(0,'shadow','Shadow'),(1,'shadow_over_water','Shadow'),(2,'water','Surface'),(3,'snow','Surface'),(4,'land','Surface'),(5,'cloud','Cloud'),(6,'flooded','Surface')],'L8':[(0,'fill','ignore'),(64,'shadow','Shadow'),(128,'clear','Surface'),(192,'thin_cloud','Cloud'),(255,'cloud','Cloud')]}
BINS=['neither','cloud_only','shadow_only','both','boundary_context_unknown'];GEOMETRY=['supported','unmatched_observed','unmatched_unknown','input_not_qualified','status_unavailable']
def locations(truth,valid):
 boundary=[]
 for cls in [1,2]:
  b=truth==cls;edge=np.zeros_like(valid)
  for axis in [0,1]:
   a=[slice(None)]*2;c=a.copy();a[axis]=slice(None,-1);c[axis]=slice(1,None);a=tuple(a);c=tuple(c);diff=(b[a]!=b[c])&valid[a]&valid[c];edge[a]|=diff;edge[c]|=diff
  boundary.append(binary_dilation(edge,structure=np.ones((5,5),bool)))
 unknown=binary_dilation(~valid,structure=np.ones((5,5),bool));unknown[:3,:]=True;unknown[-3:,:]=True;unknown[:,:3]=True;unknown[:,-3:]=True
 result=boundary[0].astype(np.uint8)+2*boundary[1].astype(np.uint8);result[unknown]=4;return result
def image_counts(pool,r):
 with open_npz(r['teacher']) as z:s=z['scores'];pred=z['pred'];v=z['valid'];pseudo=z['pseudo']
 total=s.sum(0);q=np.divide(s[2],total,out=np.zeros_like(total),where=total>1e-12);ash=v&(total>1e-12)&(pred==2)&(q>=.9);assert np.array_equal(ash,pseudo==2) and np.array_equal(pred,s.argmax(0))
 with open_npz(r['reference']) as z:y=z['truth'];valid=z['valid'];native=z['native']
 assert y.shape==native.shape==valid.shape==ash.shape==(512,512)
 if pool=='L8':assert np.array_equal(valid,native!=0)
 mapping=np.full(256,-1,np.int16)
 for i,(n,_,parent) in enumerate(NATIVE[pool]):mapping[n]=i;assert np.all(y[valid&(native==n)]=={'Surface':0,'Cloud':1,'Shadow':2,'ignore':255}[parent])
 assert np.all(mapping[native]>=0)
 geo=np.full(ash.shape,4,np.uint8)
 if r['geometry']:
  with open_npz(r['geometry']) as z:
   assert np.array_equal(ash,z['original_A_Sh'])
   if r['geometry_qualified'] is False:geo[ash]=3
   elif r['geometry_qualified'] is True:
    masks=[z['G_sun'],z['unmatched_observed'],z['unmatched_unknown']];assert np.array_equal(sum(a.astype(np.uint8) for a in masks),ash.astype(np.uint8))
    for k,m in enumerate(masks):geo[m]=k
 outcome=np.full(ash.shape,2,np.uint8);outcome[valid&(y!=2)]=0;outcome[valid&(y==2)]=1;pos=locations(y,valid);index=(((mapping[native[ash]]*5+pos[ash])*5+geo[ash])*3+outcome[ash]).astype(np.int64)
 count=np.bincount(index,minlength=len(NATIVE[pool])*5*5*3).reshape(len(NATIVE[pool]),5,5,3);assert int(count.sum())==int(ash.sum());return count
def main():
 guard();l=lock();assert not (ROOT/'ERROR_ACCEPTANCE.json').exists();composition=[];location=[];scene_summary=[];states={};checks=[]
 prior=read(P319/'PRECISION_COVERAGE.csv');strata=read(P319/'NATIVE_STRATA.csv');old={(r['pool'],r['image'],int(r['native_class'])):r for r in strata if r['selector']=='original_A_Sh'}
 for pool in ['SPARCS','L8']:
  if l['fit'][pool]['status']!='ready':states[pool]=dict(status='not_testable',reason='locked fit input or identity unavailable');continue
  aggregate={};scene_counts={}
  for r in l['fit'][pool]['rows']:
   guard();a=image_counts(pool,r);aggregate['image',r['name']]=a;scene_counts.setdefault(r['scene'],np.zeros_like(a));scene_counts[r['scene']]+=a
   for i,(code,_,_) in enumerate(NATIVE[pool]):
    previous=old.get((pool,r['name'],code));assert int(a[i].sum())==int(previous['selected']) if previous else int(a[i].sum())==0
    if previous:assert int(a[i,:,:,:2].sum())==int(previous['reference_valid'])
  aggregate.update({('scene',k):v for k,v in scene_counts.items()});aggregate['pool','POOLED']=sum(scene_counts.values());pooled=aggregate['pool','POOLED'];assert int(pooled.sum())==EXPECTED[pool]
  oldpool=next(r for r in prior if r['pool']==pool and r['level']=='pool' and r['selector']=='original_A_Sh');E,T,I=map(int,pooled.sum(axis=(0,1,2)));assert E+T==int(oldpool['reference_valid']) and T==int(oldpool['true_Shadow']) and I==int(oldpool['excluded']);assert abs(100*T/(E+T)-float(oldpool['Shadow_precision_pct']))<=1e-6
  states[pool]=dict(status='passed',A_Sh=E+T+I,E=E,T=T,invalid=I,precision_pct=100*T/(E+T),reference319_original_exact=True)
  for (level,unit),a in aggregate.items():
   errors=int(a[:,:,:,0].sum())
   for i,(code,label,parent) in enumerate(NATIVE[pool]):
    e,t,inv=map(int,a[i].sum(axis=(0,1)));composition.append(dict(pool=pool,level=level,unit=unit,native_code=code,native_class=label,parent_class=parent,accepted=e+t+inv,reference_valid=e+t,correct_Shadow=t,wrong_Shadow=e,invalid_reference=inv,share_of_all_errors_pct=100*e/errors if errors else None,precision_pct=100*t/(e+t) if e+t else None))
    for b,binname in enumerate(BINS):
     for g,gn in enumerate(GEOMETRY):
      e,t,inv=map(int,a[i,b,g]);location.append(dict(pool=pool,level=level,unit=unit,native_code=code,native_class=label,parent_class=parent,boundary_bin=binname,geometry_status=gn,accepted=e+t+inv,valid=e+t,E=e,T=t,invalid_reference=inv,share_of_all_errors_pct=100*e/errors if errors else None,precision_pct=100*t/(e+t) if e+t else None))
   if level in ['scene','pool']:
    ec={label:int(a[i,:,:,0].sum()) for i,(_,label,_) in enumerate(NATIVE[pool])};surface=sum(ec[label] for _,label,parent in NATIVE[pool] if parent=='Surface');geo={GEOMETRY[g]:int(a[:,:,g].sum()) for g in range(5)};bypos=a[:,:,:,0].sum(axis=(0,2));scene_summary.append(dict(pool=pool,level=level,scene=unit,accepted=int(a.sum()),E=errors,T=int(a[:,:,:,1].sum()),invalid_reference=int(a[:,:,:,2].sum()),water_errors=ec['water'] if pool=='SPARCS' else None,thin_cloud_errors=ec['thin_cloud'] if pool=='L8' else None,other_Cloud_errors=ec['cloud'],known_Surface_errors=surface,unresolved_Surface_errors=ec['land'] if pool=='SPARCS' else ec['clear'],near_either_boundary_E=int(bypos[1:4].sum()),far_both_boundaries_E=int(bypos[0]),boundary_context_unknown_E=int(bypos[4]),geometry_status_coverage=geo,overlap_note='water subset of known Surface; unresolved Surface subset of known Surface; location and geometry cross same E, never add descriptors; unavailable native subtype NA'))
  checks.append(dict(pool=pool,images=len(l['fit'][pool]['rows']),scenes=len(scene_counts),exact319_original_counts=True,native_strata_exact=True,partitions=True))
 if composition:
  table(ROOT/'ERROR_COMPOSITION.csv',composition);table(ROOT/'ERROR_LOCATION.csv',location);table(ROOT/'ERROR_SCENE_SUMMARY.csv',scene_summary)
 write(ROOT/'ERROR_ACCEPTANCE.json',dict(status='complete',pools=states,checks=checks,all_missing_classes_retained_zero=True,resources=guard()));print('Phase320 full fit error decomposition complete',flush=True)
if __name__=='__main__':main()
