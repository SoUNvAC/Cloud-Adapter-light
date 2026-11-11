"""Independent arithmetic, ordering, NA and historical cache-result acceptance."""
import csv,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase204/phase204_20261019'
def read(n):
 with (O/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 checks=0;signals=read('UNLABELLED_SIGNALS.csv');ranks=read('ALL_RANKINGS.csv');truth=read('ALL_GROUP_TRUTH.csv');summary=read('TABLE1_WARNING.csv');flows=read('ALL_ERROR_FLOWS.csv');seal=json.loads((O/'PRE_REFERENCE_SEAL.json').read_text())
 for n,k in [('UNLABELLED_SIGNALS.csv','signals_sha256'),('ALL_RANKINGS.csv','rankings_sha256')]:assert hashlib.sha256((O/n).read_bytes()).hexdigest()==seal[k];checks+=1
 for row in signals:
  if row['constant0']=='True' or row['constant1']=='True':assert row['RANK']==''
  checks+=1
 for pool,N,k in [('Catalogue',32,7),('PixBox',10,2)]:
  for seed in ('12201','12301'):
   tt={r['group_id']:r for r in truth if r['pool']==pool and r['seed']==seed};assert len(tt)==N
   with (R/'outputs/phase203/phase203_20261018'/f'{pool}_MECHANISM.csv').open(encoding='utf-8-sig',newline='') as f:hist={r['group_id']:r for r in csv.DictReader(f) if r['seed']==seed}
   for gid,r in tt.items():
    h=hist[gid];assert (r['delta_AP_pp']=='')==(h['11-00']=='')
    if r['delta_AP_pp']:assert abs(float(r['delta_AP_pp'])-float(h['11-00']))<1e-9;assert (r['damaged']=='True')==(float(r['delta_AP_pp'])< -1)
    if r['FPR00']:assert abs(float(r['delta_FPR'])-(float(r['FPR11'])-float(r['FPR00'])))<1e-14
    checks+=3
   for signal in ('RANK','CHANGE','ENTROPY'):
    s=[r for r in signals if r['pool']==pool and r['seed']==seed];s=sorted(s,key=lambda r:(r[signal]=='',-float(r[signal]) if r[signal] else 0,hashlib.sha256(('PHASE204_TIE_V1|'+r['group_id']).encode()).hexdigest()));rr=[r for r in ranks if r['pool']==pool and r['seed']==seed and r['signal']==signal];assert [r['group_id'] for r in rr]==[r['group_id'] for r in s]
    for i,r in enumerate(rr):assert (r['selected']=='True')==(i<k and r['value']!='');checks+=1
    selected=[r for r in rr if r['selected']=='True'];a=next(r for r in summary if r['pool']==pool and r['seed']==seed and r['signal']==signal);M=sum(r['damaged']=='True' for r in tt.values());hit=sum(tt[r['group_id']]['damaged']=='True' for r in selected);assert int(a['AP_damage_total'])==M and int(a['detected'])==hit and int(a['missed'])==M-hit;assert abs(float(a['random_expected'])-k*M/N)<1e-14;checks+=4
 for r in flows:assert sum(int(r[k]) for k in ('correct_to_wrong','wrong_to_correct','both_correct','both_wrong'))==int(r['n']);checks+=1
 with (O/'INDEPENDENT_ACCEPTANCE.json').open('x') as f:json.dump(dict(status='passed',checks=checks,historical_AP_reproduced=True,signals_seal_unchanged=True,forward=0,fit=0),f,indent=2)
 print('independent acceptance:',checks,flush=True)
if __name__=='__main__':main()
