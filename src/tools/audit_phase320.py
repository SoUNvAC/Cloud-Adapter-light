"""Independent raw-reference/cross-table count and selected-scene integer acceptance."""
from phase320_common import *
from collections import defaultdict
from fractions import Fraction
import numpy as np
from scipy.ndimage import maximum_filter
def boundary_independent(y,v):
 out=[]
 for cls in [1,2]:
  b=y==cls;horizontal=(b[:,:-1]^b[:,1:])&v[:,:-1]&v[:,1:];vertical=(b[:-1]^b[1:])&v[:-1]&v[1:];edge=np.zeros(y.shape,np.uint8);edge[:,:-1]|=horizontal;edge[:,1:]|=horizontal;edge[:-1]|=vertical;edge[1:]|=vertical;out.append(maximum_filter(edge,size=5,mode='constant',cval=0)>0)
 r,c=np.indices(y.shape);unknown=(maximum_filter((~v).astype(np.uint8),size=5,mode='constant',cval=0)>0)|(r<=2)|(c<=2)|(r>=len(y)-3)|(c>=y.shape[1]-3);bins=out[0].astype(np.uint8)+2*out[1].astype(np.uint8);bins[unknown]=4;return bins
def main():
 guard();l=lock();a=load(ROOT/'ERROR_ACCEPTANCE.json');t=load(ROOT/'THRESHOLD_ACCEPTANCE.json');checks=[]
 if (ROOT/'ERROR_LOCATION.csv').exists():
  loc=read(ROOT/'ERROR_LOCATION.csv');comp=read(ROOT/'ERROR_COMPOSITION.csv');bmap={k:i for i,k in enumerate(['neither','cloud_only','shadow_only','both','boundary_context_unknown'])};gmap={k:i for i,k in enumerate(['supported','unmatched_observed','unmatched_unknown','input_not_qualified','status_unavailable'])};byimage=defaultdict(list)
  for r in loc:
   if r['level']=='image':byimage[r['pool'],r['unit']].append(r)
  for pool in ['SPARCS','L8']:
   if l['fit'][pool]['status']!='ready':continue
   per_sc=defaultdict(lambda:np.zeros(3,np.int64))
   for row in l['fit'][pool]['rows']:
    guard()
    with open_npz(row['reference']) as z:y=z['truth'];v=z['valid'];native=z['native']
    if row['geometry']:
     with open_npz(row['geometry']) as z:
      ash=z['original_A_Sh'];g=np.full(ash.shape,4,np.uint8)
      if row['geometry_qualified']:
       g[z['G_sun']]=0;g[z['unmatched_observed']]=1;g[z['unmatched_unknown']]=2
      elif row['geometry_qualified'] is False:g[ash]=3
    else:
     with open_npz(row['teacher']) as z:ash=z['pseudo']==2
     g=np.full(ash.shape,4,np.uint8)
    b=boundary_independent(y,v);outcome=np.where(~v,2,np.where(y==2,1,0));idx=(((native[ash].astype(np.int64)*5+b[ash])*5+g[ash])*3+outcome[ash]);hist=np.bincount(idx,minlength=256*5*5*3).reshape(256,5,5,3)
    for r in byimage[pool,row['name']]:
     cc=hist[int(r['native_code']),bmap[r['boundary_bin']],gmap[r['geometry_status']]];assert cc.tolist()==[int(r['E']),int(r['T']),int(r['invalid_reference'])] and int(cc.sum())==int(r['accepted'])
    per_sc[row['scene']]+=hist.sum(axis=(0,1,2));checks.append(dict(check='independent maximum-filter boundary and every native/location/geometry outcome cell',pool=pool,name=row['name'],passed=True))
   for sc,cc in per_sc.items():
    rr=[r for r in comp if r['pool']==pool and r['level']=='scene' and r['unit']==sc];assert cc.tolist()==[sum(int(r[k]) for r in rr) for k in ['wrong_Shadow','correct_Shadow','invalid_reference']]
   rr=[r for r in comp if r['pool']==pool and r['level']=='pool'];cc=sum(per_sc.values());assert cc.tolist()==[sum(int(r[k]) for r in rr) for k in ['wrong_Shadow','correct_Shadow','invalid_reference']]
 if (ROOT/'THRESHOLD_MATCH.csv').exists():
  match=read(ROOT/'THRESHOLD_MATCH.csv');scenes=read(ROOT/'THRESHOLD_SCENE.csv')
  for r in match:
   if r['point_kind']!='selected':continue
   rr=[s for s in scenes if all(s[k]==r[k] for k in ['pool','arm','target'])];assert len(rr)==len({s['scene'] for s in l['threshold'][r['pool']]['rows']})
   for key in ['TP','FP','FN','TN']:assert sum(int(s[key]) for s in rr)==int(r[key])
   for metric,num,den in [('macro_recall_pct','TP','positive'),('macro_FPR_pct','FP','negative')]:
    values=[Fraction(int(s[num]),int(s[den])) for s in rr if int(s[den])];exact=100*float(sum(values,Fraction())/len(values));assert abs(exact-float(r[metric]))<=1e-6
   checks.append(dict(check='selected global threshold scene counts, pure-scene NA and macro means',pool=r['pool'],arm=r['arm'],target=r['target'],passed=True))
 for p,e in l['inputs'].items():assert sha(p)==e['sha256'],p
 for p,h in l['code'].items():assert sha(R/p)==h and sha(ROOT/'code'/Path(p).name)==h,p
 assert sha(ROOT/'DECISION_WINDOW.json')==l['window_sha256'];assert all(e['status']=='verified_complete_then_removed' for e in load(ROOT/'TEMP_FILES.json')) if (ROOT/'TEMP_FILES.json').exists() else True
 write(ROOT/'INDEPENDENT_QA.json',dict(status='passed',checks=checks,all_original_inputs_and_code_SHA_reverified=True,original_decision_window_unchanged=True,model_calls=0,scientific_local_work=False,resources=guard()));print('Phase320 independent counts, boundary implementation, input/code SHA passed',flush=True)
if __name__=='__main__':main()
