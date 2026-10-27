"""Independent count implementation, partitions, ranks and actual code/input integrity."""
from phase319_common import *
from collections import defaultdict
import numpy as np
def main():
 guard();lock=load(ROOT/'SELECTION_LOCK.json');quality=load(ROOT/'QUALITY_LOCK.json');assert sha(ROOT/'SELECTION_LOCK.json')==quality['selection_lock_sha256'];totals=defaultdict(lambda:np.zeros(5,np.int64));checks=[];image={(r['pool'],r['unit'],r['selector']):r for r in read(ROOT/'PRECISION_COVERAGE.csv') if r['level']=='image'};refs={(r['pool'],r['name']):r for r in quality['references']}
 for e in lock['rows']:
  guard();p,n=e['pool'],e['name'];assert sha(e['file'])==e['sha256'];rf=refs[p,n];assert sha(rf['path'])==rf['sha256']
  with np.load(e['file']) as z:m={k:z[k] for k in z.files}
  with np.load(rf['path']) as z:y=z['truth'];v=z['valid']
  m['eligible_A_Sh']=m['original_A_Sh'] if e['status']=='qualified' else np.zeros_like(v)
  assert np.all(m['G_sun']<=m['original_A_Sh']) and np.all(m['G_reverse']<=m['original_A_Sh']) and np.all(m['G_conf']<=m['original_A_Sh'])
  assert np.array_equal(m['G_sun'].astype(int)+m['unmatched_observed']+m['unmatched_unknown'],m['original_A_Sh'].astype(int));assert int(m['G_sun'].sum())==int(m['G_conf'].sum())
  teacher=next(r for r in load(PRIOR/p/'PSEUDO_LOCK.json')['rows'] if r['name']==n);assert sha(teacher['file'])==e['teacher_sha256']
  with np.load(teacher['file']) as z:s=z['scores'];den=s.sum(0);q=np.divide(s,den[None],out=np.zeros_like(s),where=den[None]>1e-12);a=z['valid']&(den>1e-12)&(s.argmax(0)==2)&(q[2]>=.9)
  assert np.array_equal(a,m['original_A_Sh']);ids=np.flatnonzero(a);order=sorted(ids.tolist(),key=lambda i:(-float(q[2].ravel()[i]),i));expected=np.zeros_like(a);expected.ravel()[order[:int(m['G_sun'].sum())] if e['status']=='qualified' else []]=True;assert np.array_equal(expected,m['G_conf'])
  for k in ['original_A_Sh','eligible_A_Sh','G_sun','G_reverse','G_conf','unmatched_observed','unmatched_unknown','supported_context_incomplete']:
   values=y[m[k]&v];hist=np.bincount(values,minlength=3);cc=np.array(list(hist[:3])+[np.count_nonzero(m[k]),np.count_nonzero(m[k]&~v)],np.int64);r=image[p,n,k];assert cc.tolist()==[int(r[f]) for f in ['true_Surface','true_Cloud','true_Shadow','selected','excluded']];totals[p,k]+=cc
  checks.append(dict(pool=p,name=n,partitions=True,all_reference_counts_exact=True,original_teacher_A_Sh_exact=True,same_k_original_q_ranking_exact=True))
 for r in read(ROOT/'PRECISION_COVERAGE.csv'):
  if r['level']=='pool':assert totals[r['pool'],r['selector']].tolist()==[int(r[f]) for f in ['true_Surface','true_Cloud','true_Shadow','selected','excluded']]
 pre=load(ROOT/'PREREGISTRATION.json')
 for rel,h in pre['code'].items():assert sha(R/rel)==h and sha(ROOT/'code'/Path(rel).name)==h
 for e in pre['dependencies']:assert sha(e['path'])==e['sha256']
 for p,e in load(ROOT/'INPUT_SHA256.json').items():assert sha(p)==e['sha256']
 source=R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)==SOURCE
 assert sha(ROOT/'DECISION_WINDOW.json')==pre['decision_window_sha256'];assert all(int(totals[p,'original_A_Sh'][3])==EXPECTED[p] for p in POOLS)
 write(ROOT/'INDEPENDENT_QA.json',dict(status='passed',products=checks,original_inputs_and_code_unchanged=True,source_sha256=SOURCE,model_imported=False,neural_calls=0,no_training=True,resources=guard()));print('Phase319 all160 independent counts/ranking/partitions/input/code hashes passed',flush=True)
if __name__=='__main__':main()
