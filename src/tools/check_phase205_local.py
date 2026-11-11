"""Independent table/confusion/AP-difference/bootstrap arithmetic acceptance."""
import csv,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase205/phase205_20261019';sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(n):
 with (O/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert abs(float(a)-float(b))<1e-9,(a,b)
def main():
 checks=0
 for pool,N,A in [('Catalogue',32,7),('PixBox',10,8)]:
  rows=read(pool+'_PER_GROUP.csv');sums=read(pool+'_SUMMARY.csv');diffs=read(pool+'_REPEATED_DAMAGE.csv');intervals=load(O/(pool+'_INTERVALS.json'))['rows'];assert len(rows)==6*N and len(sums)==6 and len(diffs)==2*N
  for s in sums:
   rr=[r for r in rows if r['strategy']==s['strategy']];assert len(rr)==N and sum(bool(r['AP']) for r in rr)==A;close(s['macro_AP_pp'],sum(float(r['AP']) for r in rr if r['AP'])*100/A)
   for field,source in [('official_decision_macro_recall_pp','recall'),('macro_negative_shadow_fraction_pp','negative_shadow_fraction')]:v=[float(r[source]) for r in rr if r[source]!=''];close(s[field],100*sum(v)/len(v));checks+=1
   mat=np.sum([np.asarray(json.loads(r['confusion']),np.int64) for r in rr],axis=0);assert mat.tolist()==load(O/(pool+'_CONFUSION.json'))['matrices'][s['strategy']]
   for r in rr:
    m=np.asarray(json.loads(r['confusion']));assert m.sum()==int(r['positive'])+int(r['negative']);assert m[2,2]==int(r['TP']) and m[2,:2].sum()==int(r['FN']) and m[:2,2].sum()==int(r['FP']);checks+=4
   if pool=='Catalogue':union=mat.sum(0)+mat.sum(1)-np.diag(mat);v=[mat[i,i]/union[i] for i in range(3) if union[i]>0];close(s['Catalogue_pooled_mIoU_pp'],100*sum(v)/len(v));checks+=1
   else:assert s['Catalogue_pooled_mIoU_pp']=='';assert int(s['positive'])==379 and int(s['negative'])==8190;checks+=3
  by={(r['strategy'],r['group_id']):r for r in rows}
  for d in diffs:
   baseline='R4' if d['family']=='member' else 'E4';prefix='C4-' if d['family']=='member' else 'E4-';a=by[baseline,d['group_id']]
   for seed in (12201,12301):
    b=by[prefix+str(seed),d['group_id']];val=d[f'seed{seed}_delta_AP_pp'];assert (val=='')==(a['AP']=='')
    if val:close(val,(float(b['AP'])-float(a['AP']))*100)
    checks+=2
   if d['baseline_AP']=='':assert d['both_damage']==''
   else:assert (d['both_damage']=='True')==(float(d['seed12201_delta_AP_pp'])< -1 and float(d['seed12301_delta_AP_pp'])< -1)
   checks+=1
  for ci in intervals:
   v=np.asarray([float(r[f"seed{ci['seed']}_delta_AP_pp"]) for r in diffs if r['family']==ci['family'] and r['baseline_AP']!=''],np.float64);assert len(v)==A;idx=np.random.default_rng(205019).integers(A,size=(10000,A));means=np.mean(v[idx],axis=1);bounds=np.quantile(means,[.025,.975]);close(ci['mean_delta_AP_pp'],np.mean(v));assert np.max(abs(bounds-ci['CI95_pp']))<1e-10;assert ci['damaged_lt_minus1pp']==int((v< -1).sum());checks+=10004
 with (O/'INDEPENDENT_FINAL_ACCEPTANCE.json').open('x') as f:json.dump(dict(status='passed',checks=checks,independent_AP_and_raw_confusion_checks='CPU_ACCEPTANCE.json',fits=0,neural_calls=0,bootstrap_seed=205019),f,indent=2)
 print('final independent checks',checks)
if __name__=='__main__':main()
