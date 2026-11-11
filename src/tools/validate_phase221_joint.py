"""Independent explicit six-axis summation; no reuse of analysis aggregation functions."""
from pathlib import Path
import json,csv,hashlib,collections,itertools,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase221/phase221_20261024';RAW=O/'delivery';CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
 clock=time.monotonic();manifest=js(RAW/'DELIVERY_MANIFEST.json');assert all((RAW/n).stat().st_size==r['bytes'] and sha(RAW/n)==r['SHA'] for n,r in manifest.items());complete=js(RAW/'COMPLETE.json');assert complete['units']==75018
 expect=js(RAW/'EXPECTED_VAL.json');support={v['name']:{**v,'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256']} for v in js(RAW/'evaluation/INPUT_SUPPORT_LOCK.json')['source']};current={f'{s}/{a}':{v['name']:v['confusion'] for v in js(RAW/f'evaluation/{s}_{a}_a1__source.json')['per_input']} for s in [21701,21702] for a in ['N','P']};acc=collections.defaultdict(collections.Counter);flowacc=collections.defaultdict(collections.Counter);perroi=collections.defaultdict(lambda:np.zeros((3,3),np.int64));allcells=0
 raw=lines(RAW/'evaluation/JOINT_COUNTS.jsonl');assert len(raw)==1070
 for r in raw:
  s=r['seed'];name=r['name'];roi=r['ROI'];v=support[name];e=expect['per_input'][name];assert r['image_SHA']==v['image_SHA'] and r['mask_SHA']==v['mask_SHA'] and r['teacher_prediction_SHA']==e['teacher_prediction_SHA'] and r['high_mask_SHA']==e['high_mask_SHA'];assert np.asarray(r['J']).shape==(2,3,3,3,3,3)
  cm={a:np.zeros((3,3),np.int64) for a in ['D','N','P']};tc={a:np.zeros((3,3,3),np.int64) for a in ['D','N','P']};bands=[np.zeros((3,3),np.int64) for _ in [0,1]];total=0
  for band,c,t,d,g,ss in itertools.product(range(2),range(3),range(3),range(3),range(3),range(3)):
   n=r['J'][band][c][t][d][g][ss];assert isinstance(n,int) and n>=0;total+=n;allcells+=1
   if not n:continue
   bands[band][c,t]+=n
   for a,p in [('D',d),('N',g),('P',ss)]:cm[a][c,p]+=n;tc[a][c,t,p]+=n
   for region in [roi,'__pooled__']:
    for tier in ['all','correct' if t==c else 'wrong']:
     q=acc[s,region,band,c,tier];q['valid_pixels']+=n;q['D_errors']+=n*(d!=c);q['N_errors']+=n*(g!=c);q['P_errors']+=n*(ss!=c);q['D_wrong_N_right']+=n*(d!=c and g==c);q['D_right_N_wrong']+=n*(d==c and g!=c)
     for dest in ([0,1] if c==2 else [2]):
      f=flowacc[s,region,band,c,tier,dest];f['valid_reference_pixels']+=n;f['D_flow_pixels']+=n*(d==dest);f['N_flow_pixels']+=n*(g==dest);f['P_flow_pixels']+=n*(ss==dest);f['D_flow_removed_by_N']+=n*(d==dest and g!=dest);f['N_flow_added_from_other_D_predictions']+=n*(d!=dest and g==dest)
  assert total==r['valid_pixels'] and total==262144
  for a in ['D','N','P']:
   if a=='D':
    assert cm[a].tolist()==e['students'][f'{s}/D']['confusion'] and tc[a].tolist()==e['students'][f'{s}/D']['reference_teacher_student_counts'];assert r['student_prediction_SHA']['D']==e['students'][f'{s}/D']['prediction_SHA']
   else:assert cm[a].tolist()==current[f'{s}/{a}'][name]
   perroi[s,a,roi]+=cm[a]
  for b,band in enumerate(['high','low']):assert bands[b].tolist()==e['bands'][band]['confusion']
 for table in ['TABLE_C_POOLED.csv','TABLE_C_ALL_ROIS.csv']:
  for r in csvrows(O/table):
   s=int(r['seed']);key=s,r['ROI'],['high','low'].index(r['band']),CS.index(r['reference_class']),r['teacher_reference_correctness'];v=acc[key]
   for k in ['valid_pixels','D_errors','N_errors','P_errors','D_wrong_N_right','D_right_N_wrong']:assert int(r[k])==v[k],(key,k)
   assert int(r['N_minus_D_errors'])==v['N_errors']-v['D_errors']==v['D_right_N_wrong']-v['D_wrong_N_right'];assert int(r['P_minus_D_errors'])==v['P_errors']-v['D_errors']
   for a in ['D','N','P']:
    if v['valid_pixels']:assert abs(float(r[a+'_error_rate'])-v[a+'_errors']/v['valid_pixels'])<1e-12
    else:assert r[a+'_error_rate']==''
 for r in csvrows(O/'DIRECT_ERROR_FLOWS.csv'):
  dest=CS.index(r['flow'].split('->')[1]);key=int(r['seed']),r['ROI'],['high','low'].index(r['band']),CS.index(r['reference_class']),r['teacher_reference_correctness'],dest;v=flowacc[key]
  for k in ['valid_reference_pixels','D_flow_pixels','N_flow_pixels','P_flow_pixels','D_flow_removed_by_N','N_flow_added_from_other_D_predictions']:assert int(r[k])==v[k]
  assert int(r['N_minus_D_flow'])==v['N_flow_pixels']-v['D_flow_pixels']==v['N_flow_added_from_other_D_predictions']-v['D_flow_removed_by_N']
 # All direct N/P transitions are independently enumerated from the same co-observed J.
 npacc=collections.defaultdict(collections.Counter)
 for r in raw:
  for band,c,t,d,n,p in itertools.product(range(2),range(3),range(3),range(3),range(3),range(3)):
   count=r['J'][band][c][t][d][n][p]
   if not count:continue
   for roi in [r['ROI'],'__pooled__']:
    for tier in ['all','correct' if t==c else 'wrong']:
     q=npacc[r['seed'],roi,band,c,tier];q['valid_pixels']+=count;q['P_wrong_N_right']+=count*(p!=c and n==c);q['P_right_N_wrong']+=count*(p==c and n!=c);q['N_errors']+=count*(n!=c);q['P_errors']+=count*(p!=c)
 for r in csvrows(O/'DIRECT_N_MINUS_P.csv'):
  q=npacc[int(r['seed']),r['ROI'],['high','low'].index(r['band']),CS.index(r['reference_class']),r['teacher_reference_correctness']]
  for key in ['valid_pixels','P_wrong_N_right','P_right_N_wrong','N_errors','P_errors']:assert int(r[key])==q[key]
  assert int(r['N_minus_P_errors'])==q['N_errors']-q['P_errors']==q['P_right_N_wrong']-q['P_wrong_N_right']
 allocation=collections.defaultdict(collections.Counter)
 for r in lines(RAW/'evaluation/VAL_WEIGHT_DIAGNOSTIC.jsonl'):
  w=r['weight_counts'];proof=r['weight_proof'];e=expect['per_input'][r['name']]
  for ref,t,gh,ph in itertools.product(range(3),range(3),range(2),range(2)):
   count=w[ref][t][gh][ph];assert count>=0
   if not count:continue
   q=proof['classes'][t];mean=q['mean_g'];nw=float(np.float32((.5+gh)/mean));pw=float(np.float32((.5+ph)/mean))
   for basis,c in [('reference',ref),('teacher_prediction',t)]:
    a=allocation['source_val535','unique535',basis,c];a['support']+=count;a['G_sum']+=count*(.5+gh);a['N_sum']+=count*nw;a['P_sum']+=count*pw;a['G_high']+=count*gh
  for ref,t,gh in itertools.product(range(3),range(3),range(2)):assert sum(w[ref][t][gh])==e['bands']['high' if gh else 'low']['confusion'][ref][t]
  for t in range(3):
   support=sum(w[ref][t][gh][ph] for ref,gh,ph in itertools.product(range(3),range(2),range(2)));high=sum(w[ref][t][1][ph] for ref,ph in itertools.product(range(3),range(2)));phigh=sum(w[ref][t][gh][1] for ref,gh in itertools.product(range(3),range(2)));q=proof['classes'][t];assert support==q['support'] and high==phigh
   if support:assert q['mean_g']==(.5*support+high)/support and abs(q['N_mean']-1)<=1e-6 and abs(q['P_mean']-1)<=1e-6
   else:assert q['mean_g'] is None
 freq=collections.Counter(v['source_sample_id'] for v in js(RAW/'SOURCE_SEQUENCE_LOCK.json'));assert len(freq)==256 and sum(freq.values())==4000
 for r in js(RAW/'weight_preparation/WEIGHT_MANIFEST.json'):
  assert r['repeat_count']==freq[r['sample_id']]
  for q in r['classes']:
   if not q['support']:continue
   for rep,m in [('unique256',1),('original4000_sequence',freq[r['sample_id']])]:
    a=allocation['replay256',rep,'teacher_prediction',q['class']];a['support']+=m*q['support'];a['G_sum']+=m*(.5*q['support']+q['g_high_pixels']);a['G_high']+=m*q['g_high_pixels'];a['N_sum']+=m*q['N_sum'];a['P_sum']+=m*q['P_sum']
 for r in csvrows(O/'CLASS_WEIGHT_DISTRIBUTIONS.csv'):
  a=allocation[r['pool'],r['repetition'],r['basis'],CS.index(r['class'])]
  for key in ['support','G_sum','N_sum','P_sum','G_high']:assert abs(float(r[key])-a[key])<1e-6
  for arm in ['G','N','P']:assert abs(float(r[arm+'_mean_weight'])-a[arm+'_sum']/a['support'])<1e-12
 result={'status':'passed','raw_artifacts_SHA_checked':len(manifest),'explicit_six_axis_cells':allcells,'all_source_val_D_legacy_SHA_and_margins':True,'all_new_N_P_margins_match_formal_evaluation':True,'direct_D_N_and_P_N_transitions':True,'Source_teacher_high_low_masks_reproduced':True,'val_reference_and_teacher_class_weights_reconstructed':True,'unique256_and4000_sequence_class_weights':True,'new_model_calls':0,'seconds':time.monotonic()-clock,'code_SHA':sha(Path(__file__))};(O/'INDEPENDENT_JOINT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
