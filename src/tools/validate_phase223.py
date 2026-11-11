"""Explicit four-axis count loops and independent scalar KL reconstruction."""
from pathlib import Path
import collections,csv,json,hashlib,math,time
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase223/phase223_20261024';D=O/'delivery';CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def score(c):
 c=np.asarray(c);v=[]
 for i in range(3):
  u=sum(c[i])+sum(c[:,i])-c[i,i]
  if u:v.append(c[i,i]/u)
 return 100*sum(v)/len(v)
def main():
 clock=time.monotonic();m=js(D/'DELIVERY_MANIFEST.json');assert all((D/n).stat().st_size==v['bytes'] and sha(D/n)==v['SHA'] for n,v in m.items());assert js(D/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed';complete=js(D/'COMPLETE.json');assert complete['total_units']==5360 and complete['new_fits']==0 and complete['GPU_seconds']<3600
 lock=js(D/'INPUT_MODEL_LOCK.json');expected={k:js(D/'expected'/f'{k}.json') for k in ['Source']+list(lock['models'])};byexp={k:{v['name']:v for v in q['per_input']} for k,q in expected.items()};data=lines(D/'PER_IMAGE.jsonl');assert len(data)==4280;by={(v['model'],v['name']):v for v in data};assert len(by)==4280;roi=collections.defaultdict(list);modelrows=collections.defaultdict(list);teacher_hash=collections.defaultdict(set)
 for v in data:
  key=v['model'];roi[key,v['ROI']].append(v);modelrows[key].append(v);teacher_hash[v['name']].add(v['teacher_prediction_SHA']);assert v['alpha']==.25 and v['pool']=='Source_val535' and v['KL_min']>=-1e-6 and math.isfinite(v['KL_sum_geometry']);n=v['geometry_pixels'];assert n==262144 and abs(v['KL_sum_geometry']/n-v['KL_mean_geometry'])<1e-14;assert v['negative_KL_sum']<=0
  j=np.asarray(v['reference_teacher_student_counts'],np.int64);tcm=j.sum(2);scm=j.sum(1);assert tcm.tolist()==byexp['Source'][v['name']]['confusion'] and scm.tolist()==byexp[key][v['name']]['confusion'];valid=v['reference_valid'];assert valid['pixels']==j.sum() and valid['pixels']+v['reference_ignore_pixels']==n and valid['student_errors']==scm.sum()-np.trace(scm) and valid['teacher_errors']==tcm.sum()-np.trace(tcm)
  for q in v['reference_classes']:
   c=q['class'];x=j[c];assert q['pixels']==x.sum() and q['student_errors']==x.sum()-x[:,c].sum() and q['teacher_errors']==x.sum()-x[c].sum()
  for q in v['correctness']:
   total=sum(int(j[r,t,s]) for r in range(3) for t in range(3) for s in range(3) if (r==t)==q['teacher_correct'] and (r==s)==q['student_correct']);assert q['pixels']==total
  for q in v['error_flows']:assert q['pixels']==scm[q['reference'],q['student']] and (q['KL_mean'] is None if q['pixels']==0 else abs(q['KL_mean']-q['KL_sum']/q['pixels'])<1e-14)
  for q in v['teacher_correct_shadow_destinations']:assert q['pixels']==j[2,2,q['destination']] and (q['teacher_confidence_mean'] is None if not q['pixels'] else abs(q['teacher_confidence_mean']-q['teacher_confidence_sum']/q['pixels'])<1e-12)
  for label in ['reference_classes','correctness']:assert sum(q['pixels'] for q in v[label])==valid['pixels'] and abs(sum(q['KL_sum'] for q in v[label])-valid['KL_sum'])<1e-8
 assert len(teacher_hash)==535 and all(len(v)==1 for v in teacher_hash.values());assert all(len(v)==535 for v in modelrows.values()) and len(roi)==856
 # Explicit all81 joint cells; compare separate student statistics and old cache margins.
 joint=collections.defaultdict(lambda:np.zeros((3,3,3,3),np.int64));cells=0
 for v in lines(D/'JOINT_COUNTS.jsonl'):
  j=np.asarray(v['J'],np.int64);assert j.shape==(3,3,3,3) and (j>=0).all() and j.sum()==v['valid_pixels'];trc=np.zeros((3,3,3),np.int64);trd=np.zeros_like(trc)
  for r in range(3):
   for t in range(3):
    for c in range(3):
     for d in range(3):n=int(j[r,t,c,d]);trc[r,t,c]+=n;trd[r,t,d]+=n;cells+=1
  for arm,tr in [('C',trc),('D',trd)]:
   key=f"{v['target']}_{v['seed']}_{arm}";record=by[key,v['name']];assert tr.tolist()==record['reference_teacher_student_counts'] and record['student_prediction_SHA']==v[arm+'_prediction_SHA'] and record['teacher_prediction_SHA']==v['teacher_prediction_SHA']
  joint[v['target'],v['seed'],v['ROI']]+=j
 ids=sorted(expected['Source']['per_group']);assert len(ids)==107 and cells==173340
 for row in rows('TABLE_C_CORRECTNESS.csv'):
  target,s,g=row['target'],int(row['seed']),row['ROI'];j=sum(joint[target,s,h] for h in ids) if g=='__pooled__' else joint[target,s,g];fewererrors=fewercorrections=net=0
  for r in range(3):
   for t in range(3):
    for c in range(3):
     for d in range(3):
      n=int(j[r,t,c,d]);net+=n*((d!=r)-(c!=r));fewererrors+=n*((c!=r)-(d!=r))*(t==r);fewercorrections+=n*((c==r)-(d==r))*(t!=r)
  assert net==fewercorrections-fewererrors and int(row['teacher_correct_D_fewer_errors'])==fewererrors and int(row['teacher_wrong_D_fewer_corrections'])==fewercorrections and int(row['D_minus_C_errors'])==net
 for row in rows('DIRECT_C_TO_D_TRANSITIONS.csv'):
  target,s,g=row['target'],int(row['seed']),row['ROI'];j=sum(joint[target,s,h] for h in ids) if g=='__pooled__' else joint[target,s,g];r=CS.index(row['reference_class']);sums=collections.Counter()
  for t in range(3):
   if row['teacher_correctness']=='correct' and t!=r or row['teacher_correctness']=='wrong' and t==r:continue
   for c in range(3):
    for d in range(3):n=int(j[r,t,c,d]);sums['pixels']+=n;sums['C_wrong_D_right']+=n*(c!=r and d==r);sums['C_right_D_wrong']+=n*(c==r and d!=r);sums['D_minus_C_errors']+=n*((d!=r)-(c!=r))
  assert all(int(row[k])==v for k,v in sums.items())
 # Independent scalar softmax, exp/log and class sum at fixed48 coordinates.
 maxerr=0.;coord=js(D/'FIXED_KL_COORDINATES.json');assert len(coord)==48
 for v in coord:
  a=v['teacher_S'];b=v['student_S'];ea=[math.exp(x-max(a)) for x in a];eb=[math.exp(x-max(b)) for x in b];p=[x/sum(ea) for x in ea];q=[x/sum(eb) for x in eb];kl=sum(p[i]*math.log(p[i]/q[i]) for i in range(3));err=abs(kl-v['KL_GPU']);maxerr=max(maxerr,err);assert err<=1e-6;assert abs(sum(v['p'][i]*(v['logp'][i]-v['logq'][i]) for i in range(3))-v['KL_GPU'])<5e-8
 # Margins and focus/audit reproduction, all other ROIs remain in report.
 A={v['model']:v for v in rows('TABLE_A_ALL_SOURCE.csv')}
 for key,rr in modelrows.items():
  old=expected[key];source=expected['Source'];n=sum(v['geometry_pixels'] for v in rr);ks=sum(v['KL_sum_geometry'] for v in rr);assert abs(float(A[key]['KL_mean_geometry'])-ks/n)<1e-14;losses=sorted([score(source['per_group'][g])-score(old['per_group'][g]) for g in ids],reverse=True);assert abs(float(A[key]['T11_pp'])-sum(losses[:11])/11)<1e-10 and abs(float(A[key]['source_pool_loss_pp'])-(score(source['confusion'])-score(old['confusion'])))<1e-10
 for row in rows('ALL_ROI_PAIRED_KL_ERRORS.csv'):
  target,s,g=row['target'],int(row['seed']),row['ROI'];ck=f'{target}_{s}_C';dk=f'{target}_{s}_D';cc=modelrows[ck] if g=='__pooled__' else roi[ck,g];dd=modelrows[dk] if g=='__pooled__' else roi[dk,g];cn=sum(v['geometry_pixels'] for v in cc);dn=sum(v['geometry_pixels'] for v in dd);ckl=sum(v['KL_sum_geometry'] for v in cc)/cn;dkl=sum(v['KL_sum_geometry'] for v in dd)/dn;ce=sum(v['reference_valid']['student_errors'] for v in cc);de=sum(v['reference_valid']['student_errors'] for v in dd);assert abs(float(row['C_KL'])-ckl)<1e-14 and abs(float(row['D_KL'])-dkl)<1e-14 and int(row['D_minus_C_errors'])==de-ce and (row['D_KL_lower_errors_more']=='True')==(dkl<ckl and de>ce)
 for v in lines(D/'SIGNED_GAP_SUMMARY.jsonl'):
  key,g=v['model'],v['ROI'];rr=modelrows[key] if g=='__pooled__' else roi[key,g];count=sum(next(x['pixels'] for x in r['teacher_correct_shadow_destinations'] if x['destination']==0) for r in rr)
  for kind in ['teacher','student']:
   q=v['stats'][kind];assert q['pixels']==count
   if not count:assert q['P10'] is q['median'] is q['P90'] is None
   else:assert q['P10']<=q['median']<=q['P90']
 for v in js(D/'SIGNED_GAP_FIXED_SAMPLES.json'):
  for kind in ['teacher','student']:assert float(np.float32(np.float32(v[kind+'_S'][2])-np.float32(v[kind+'_S'][0])))==v[kind+'_gap']
 audits=list(csv.DictReader((D/'audit/CASE_SUMMARY.csv').open(encoding='utf-8-sig')))
 for row in audits:
  if row['context']!='matched_alpha025':continue
  v=A[f"{row['target']}_{row['seed']}_D"];assert abs(float(row['T11_pp'])-float(v['T11_pp']))<1e-10
 for row in csv.DictReader((D/'audit/L8_SELECTED_WORST_ROI_CROSS_TARGET.csv').open(encoding='utf-8-sig')):
  if row['context']!='matched_alpha025':continue
  key=f"{row['target']}_{row['seed']}_D";g=row['ROI'];old=expected[key]['per_group'][g];assert abs(score(old)-float(row['D_mIoU']))<1e-10 and abs(score(expected['Source']['per_group'][g])-score(old)-float(row['loss_pp']))<1e-10
 frozen=js(D/'FROZEN_FINAL.json');assert frozen['initial_digests']==frozen['final_digests'] and frozen['all_no_grad_eval'];gate=js(D/'ENGINEERING_GATE_ACCEPTED.json');assert gate['nine_official_wrappers_semantic_bitwise_equal'] and gate['formal_first_input_reused'] and abs(gate['Source_clone_KL_min'])<=1e-6 and abs(gate['Source_clone_KL_max'])<=1e-6
 result={'status':'passed','raw_files_SHA_checked':len(m),'all8_old_alpha025_confusions_predictions_exact':True,'all107_ROIs_each_target_seed':True,'explicit_four_axis_cells':cells,'independent_scalar_KL_coordinates':len(coord),'max_CPU_GPU_KL_abs_error':maxerr,'tolerance':1e-6,'all_correctness_identities_and_direct_transitions':True,'ROI0124_and_T11_local_audit_reproduced':True,'signed_gap_support_and_fixed_samples_checked':True,'full_gap_quantiles':'computed over all selected float32 output gaps with CPU linear quantiles; not independently replayed from pixels','new_calls_or_training_or_bootstrap':0,'seconds':time.monotonic()-clock,'code_SHA':sha(Path(__file__))};(O/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
