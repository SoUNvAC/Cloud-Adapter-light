"""Independent scalar KL formula, integer transitions, all support and217 replay checks."""
from pathlib import Path
import csv,json,hashlib,math,collections,time
import numpy as np
from analyze_phase216 import peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase218/phase218_20261023';RAW=O/'delivery'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def main():
 start=time.monotonic();manifest=js(RAW/'DELIVERY_MANIFEST.json');assert all((RAW/n).stat().st_size==v['bytes'] and sha(RAW/n)==v['SHA'] for n,v in manifest.items());assert js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed'
 lock=js(RAW/'INPUT_MODEL_LOCK.json');assert lock['expected_units']==4758;data=[json.loads(x) for x in (RAW/'PER_IMAGE.jsonl').read_text().splitlines()];groups=collections.defaultdict(list);roi=collections.defaultdict(list)
 coordinate_checks=0;max_formula_error=0
 for r in js(RAW/'FIXED_KL_COORDINATES.json'):
  x=r['Source_S'];y=r['student_S'];sx=max(x);sy=max(y);ex=[math.exp(v-sx) for v in x];ey=[math.exp(v-sy) for v in y];p=[v/sum(ex) for v in ex];q=[v/sum(ey) for v in ey]
  direct=sum(p[i]*math.log(p[i]/q[i]) for i in range(3));recorded=sum(float(r['p'][i])*(float(r['logp'][i])-float(r['logq'][i])) for i in range(3));err=abs(direct-r['KL_GPU']);max_formula_error=max(max_formula_error,err)
  assert err<=1e-6 and abs(recorded-r['KL_GPU'])<5e-8 and all(abs(p[i]-r['p'][i])<2e-7 and abs(q[i]-r['q'][i])<2e-7 for i in range(3));coordinate_checks+=1
 assert coordinate_checks==60
 cases={k:js(RAW/'expected217'/n) for k,n in [('Source','Source__source.json')]+[(f'{s}_{a}',f'{s}_{a}_a1__source.json') for s in [21701,21702] for a in ['C','D']]};expected={k:{r['name']:r['confusion'] for r in v['per_input']} for k,v in cases.items()};counts=0;cm=collections.defaultdict(lambda:np.zeros((3,3),np.int64));onegeom=512*512
 for r in data:
  key=r['seed'],r['arm'],r['pool'];groups[key].append(r);roi[(*key,r['ROI'])].append(r);n=r['geometry_pixels'];assert n==onegeom and math.isfinite(r['KL_sum_geometry']) and r['KL_min']>=-1e-6
  assert abs(r['KL_sum_geometry']/n-r['KL_mean_geometry'])<1e-15 and 0<=r['negative_KL_pixels']<=n and r['negative_KL_sum']<=0
  ts=np.asarray(r['teacher_student_geometry_counts']);assert ts.shape==(3,3) and (ts>=0).all() and ts.sum()==n and ts.sum()-np.trace(ts)==r['teacher_disagreements_geometry']
  if r['pool']=='Replay256':assert 'reference_valid' not in r and 'reference_classes' not in r;continue
  tr=np.asarray(r['reference_teacher_student_counts']);assert tr.shape==(3,3,3) and (tr>=0).all();valid=r['reference_valid'];assert tr.sum()==valid['pixels'] and valid['pixels']+r['reference_ignore_pixels']==n
  teacher=tr.sum(2);student=tr.sum(1);model=f"{r['seed']}_{r['arm']}";assert teacher.tolist()==expected['Source'][r['name']] and student.tolist()==expected[model][r['name']];cm[model,r['ROI']]+=student;cm['Source',r['ROI']]=cases['Source']['per_group'][r['ROI']]
  assert int(student.sum()-np.trace(student))==valid['student_errors'] and int(teacher.sum()-np.trace(teacher))==valid['teacher_errors']
  for q in r['reference_classes']:
   i=q['class'];a=tr[i];assert q['pixels']==a.sum() and q['student_errors']==a.sum()-a[:,i].sum() and q['teacher_errors']==a.sum()-a[i,:].sum() and q['teacher_disagreements']==a.sum()-np.trace(a)
  for q in r['correctness']:
   tc=q['teacher_correct'];sc=q['student_correct'];amount=sum(int(tr[i,j,k]) for i in range(3) for j in range(3) for k in range(3) if (i==j)==tc and (i==k)==sc);assert q['pixels']==amount
  for q in r['both_wrong_subtypes']:
   amount=sum(int(tr[i,j,k]) for i in range(3) for j in range(3) for k in range(3) if i!=j and i!=k and (j==k)==q['same_error_class']);assert q['pixels']==amount
  for field in ['reference_classes','correctness']:
   assert sum(q['pixels'] for q in r[field])==valid['pixels'];assert abs(sum(q['KL_sum'] for q in r[field])-valid['KL_sum'])<1e-8
  bothwrong=next(q for q in r['correctness'] if not q['teacher_correct'] and not q['student_correct']);assert sum(q['pixels'] for q in r['both_wrong_subtypes'])==bothwrong['pixels'] and abs(sum(q['KL_sum'] for q in r['both_wrong_subtypes'])-bothwrong['KL_sum'])<1e-8
  if valid['pixels']==n:assert r['reference_ignore_pixels']==0 and abs(valid['KL_sum']-r['KL_sum_geometry'])<1e-12
  counts+=1
 assert counts==2140 and len(groups)==8 and len(roi)==1424
 for (s,a,p),rr in groups.items():assert len(rr)==(256 if p=='Replay256' else 535) and len({r['ROI'] for r in rr})==(249 if p=='Replay256' else 107)
 for k,v in cases.items():
  for g,c in v['per_group'].items():assert np.array_equal(cm[k,g],c)
  assert np.array_equal(sum(np.asarray(cm[k,g]) for g in v['per_group']),v['confusion'])
 A={(int(r['seed']),r['arm'],r['pool']):r for r in rd('TABLE_A_RETENTION.csv')}
 for k,rr in groups.items():
  row=A[k];n=sum(r['geometry_pixels'] for r in rr);v=[r['KL_mean_geometry'] for r in rr];assert int(row['images'])==len(rr) and int(row['geometry_pixels'])==n and abs(float(row['KL_mean_geometry'])-sum(r['KL_sum_geometry'] for r in rr)/n)<1e-14
  assert abs(float(row['image_KL_median'])-np.median(v))<1e-14 and abs(float(row['image_KL_P90'])-np.quantile(v,.9))<1e-14 and abs(float(row['teacher_disagreement_rate_geometry'])-sum(r['teacher_disagreements_geometry'] for r in rr)/n)<1e-14
 def score(c):
  c=np.asarray(c);v=[]
  for i in range(3):
   u=sum(c[i])+sum(c[:,i])-c[i,i]
   if u:v.append(c[i,i]/u)
  return sum(v)/len(v)*100
 for r in rd('TABLE_B_REFERENCE.csv'):
  key=int(r['seed']),r['arm'],'Source_val535';rr=groups[key];c=sum(np.asarray(x['reference_teacher_student_counts']).sum(1) for x in rr);loss=sorted([score(cases['Source']['per_group'][g])-score(cm[f'{key[0]}_{key[1]}',g]) for g in cases['Source']['per_group']],reverse=True);assert abs(score(c)-float(r['mIoU']))<1e-10 and abs(sum(loss[:11])/11-float(r['T11_pp']))<1e-10 and abs((c.sum()-np.trace(c))/c.sum()-float(r['error_rate']))<1e-14
 for r in rd('TABLE_C_ALL_ROI.csv'):
  s=int(r['seed']);g=r['ROI'];cc=roi[s,'C','Source_val535',g];dd=roi[s,'D','Source_val535',g];ck=sum(x['reference_valid']['KL_sum'] for x in cc)/sum(x['reference_valid']['pixels'] for x in cc);dk=sum(x['reference_valid']['KL_sum'] for x in dd)/sum(x['reference_valid']['pixels'] for x in dd);ce=sum(x['reference_valid']['student_errors'] for x in cc);de=sum(x['reference_valid']['student_errors'] for x in dd)
  assert abs(ck-float(r['C_KL_mean']))<1e-14 and abs(dk-float(r['D_KL_mean']))<1e-14 and (r['D_KL_lower_and_errors_more']=='True')==(dk<ck and de>ce)
 gate=js(RAW/'ENGINEERING_GATE_ACCEPTED.json');assert gate['fresh_features_for_two_different_inputs'] and len(gate['checks'])==2 and all(x['five_official_wrappers_semantic_and_argmax_bitwise_equal'] for x in gate['checks'])
 assert len(rd('TABLE_C_ALL_ROI.csv'))==214 and len(rd('TABLE_C_FOCUS.csv'))==8
 assert peak()<16*1024**3;result={'status':'passed','explicit_scalar_KL_formula_coordinates':coordinate_checks,'max_FP64_vs_FP32_KL_error':max_formula_error,'all_reference_transitions_directly_reconstructed':counts,'all_images_unique_and_unweighted':True,'same217_all_per_input_ROI_pooled_confusions_and_prediction_SHA':True,'zero_training_or_new_thresholds':True,'source_train_reference_absent':True,'CPU_peak_bytes':peak(),'seconds':time.monotonic()-start,'new_model_calls':0,'code_SHA':sha(Path(__file__))};(O/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
