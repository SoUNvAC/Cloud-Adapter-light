"""Independent explicit six-axis summation; no reuse of analysis aggregation functions."""
from pathlib import Path
import json,csv,hashlib,collections,itertools,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase220/phase220_20261024';RAW=O/'delivery';CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
 clock=time.monotonic();manifest=js(RAW/'DELIVERY_MANIFEST.json');assert all((RAW/n).stat().st_size==r['bytes'] and sha(RAW/n)==r['SHA'] for n,r in manifest.items());complete=js(RAW/'COMPLETE.json');assert complete['units']==4800 and complete['training_steps']==complete['fits']==complete['backwards']==complete['target_inference']==0
 expect=js(RAW/'EXPECTED_VAL.json');sup=js(RAW/'SUPPORT_LOCK.json');support={v['name']:v for v in sup['val']};acc=collections.defaultdict(collections.Counter);flowacc=collections.defaultdict(collections.Counter);perroi=collections.defaultdict(lambda:np.zeros((3,3),np.int64));allcells=0
 raw=lines(RAW/'val/JOINT_COUNTS.jsonl');assert len(raw)==1070
 for r in raw:
  s=r['seed'];name=r['name'];roi=r['ROI'];v=support[name];e=expect['per_input'][name];assert r['image_SHA']==v['image_SHA'] and r['mask_SHA']==v['mask_SHA'] and r['teacher_prediction_SHA']==e['teacher_prediction_SHA'] and r['high_mask_SHA']==e['high_mask_SHA'];assert r['shape']==[2,3,3,3,3,3]
  cm={a:np.zeros((3,3),np.int64) for a in ['D','G','S']};tc={a:np.zeros((3,3,3),np.int64) for a in ['D','G','S']};bands=[np.zeros((3,3),np.int64) for _ in [0,1]];total=0
  for band,c,t,d,g,ss in itertools.product(range(2),range(3),range(3),range(3),range(3),range(3)):
   n=r['J'][band][c][t][d][g][ss];assert isinstance(n,int) and n>=0;total+=n;allcells+=1
   if not n:continue
   bands[band][c,t]+=n
   for a,p in [('D',d),('G',g),('S',ss)]:cm[a][c,p]+=n;tc[a][c,t,p]+=n
   for region in [roi,'__pooled__']:
    for tier in ['all','correct' if t==c else 'wrong']:
     q=acc[s,region,band,c,tier];q['valid_pixels']+=n;q['D_errors']+=n*(d!=c);q['G_errors']+=n*(g!=c);q['S_errors']+=n*(ss!=c);q['D_wrong_G_right']+=n*(d!=c and g==c);q['D_right_G_wrong']+=n*(d==c and g!=c)
     for dest in ([0,1] if c==2 else [2]):
      f=flowacc[s,region,band,c,tier,dest];f['valid_reference_pixels']+=n;f['D_flow_pixels']+=n*(d==dest);f['G_flow_pixels']+=n*(g==dest);f['S_flow_pixels']+=n*(ss==dest);f['D_flow_removed_by_G']+=n*(d==dest and g!=dest);f['G_flow_added_from_other_D_predictions']+=n*(d!=dest and g==dest)
  assert total==r['valid_pixels'] and total+r['ignore_pixels']==262144
  for a in ['D','G','S']:
   assert cm[a].tolist()==e['students'][f'{s}/{a}']['confusion'] and tc[a].tolist()==e['students'][f'{s}/{a}']['reference_teacher_student_counts'];perroi[s,a,roi]+=cm[a]
   if 'prediction_SHA' in e['students'][f'{s}/{a}']:assert r['student_prediction_SHA'][a]==e['students'][f'{s}/{a}']['prediction_SHA']
  for b,band in enumerate(['high','low']):assert bands[b].tolist()==e['bands'][band]['confusion']
 for table in ['TABLE_A_POOLED.csv','TABLE_A_ALL_ROIS.csv']:
  for r in csvrows(O/table):
   s=int(r['seed']);key=s,r['ROI'],['high','low'].index(r['band']),CS.index(r['reference_class']),r['teacher_reference_correctness'];v=acc[key]
   for k in ['valid_pixels','D_errors','G_errors','S_errors','D_wrong_G_right','D_right_G_wrong']:assert int(r[k])==v[k],(key,k)
   assert int(r['G_minus_D_errors'])==v['G_errors']-v['D_errors']==v['D_right_G_wrong']-v['D_wrong_G_right'];assert int(r['S_minus_D_errors'])==v['S_errors']-v['D_errors']
   for a in ['D','G','S']:
    if v['valid_pixels']:assert abs(float(r[a+'_error_rate'])-v[a+'_errors']/v['valid_pixels'])<1e-12
    else:assert r[a+'_error_rate']==''
 for r in csvrows(O/'DIRECT_ERROR_FLOWS.csv'):
  dest=CS.index(r['flow'].split('->')[1]);key=int(r['seed']),r['ROI'],['high','low'].index(r['band']),CS.index(r['reference_class']),r['teacher_reference_correctness'],dest;v=flowacc[key]
  for k in ['valid_reference_pixels','D_flow_pixels','G_flow_pixels','S_flow_pixels','D_flow_removed_by_G','G_flow_added_from_other_D_predictions']:assert int(r[k])==v[k]
  assert int(r['G_minus_D_flow'])==v['G_flow_pixels']-v['D_flow_pixels']==v['G_flow_added_from_other_D_predictions']-v['D_flow_removed_by_G']
 # Reconstruct the previous219 per-ROI class audit without re-opening any pixels.
 prior=ROOT/'outputs/research_audits/phase219_class_damage_20261024';oldaudit=js(RAW/'ORIGINAL_CLASS_AUDIT_SHA.json');assert all(sha(prior/n)==v for n,v in oldaudit.items())
 for r in csvrows(prior/'PER_ROI_REFERENCE_CLASS.csv'):
  # Historical schema is checked below before any comparison.
  s=int(r['seed']);roi=r['ROI'];c=CS.index(r['reference_class']);d=perroi[s,'D',roi];g=perroi[s,'G',roi];assert int(r['G_minus_D_reference_errors'])==int((g[c].sum()-g[c,c])-(d[c].sum()-d[c,c]))
  correct=sum(acc[s,roi,bi,c,'correct']['G_errors']-acc[s,roi,bi,c,'correct']['D_errors'] for bi in [0,1]);wrong=sum(acc[s,roi,bi,c,'wrong']['G_errors']-acc[s,roi,bi,c,'wrong']['D_errors'] for bi in [0,1]);assert -correct==int(r['teacher_correct_preservation_gain']) and wrong==int(r['teacher_error_corrections_lost']);assert int(r['support'])==int(d[c].sum())
 wb={};wb_roi={};seen=set()
 def add(pool,basis,method,cl,v,mult=1,roi=None):
  key=pool,basis,method,cl;accum=wb.setdefault(key,collections.Counter())
  for k in ['support','G_high_pixels','S_high_pixels','G_weight_sum','S_weight_sum']:accum[k]+=v[k]*mult
  if roi is not None:
   accum=wb_roi.setdefault(key+(roi,),collections.Counter())
   for k in ['support','G_high_pixels','S_high_pixels','G_weight_sum','S_weight_sum']:accum[k]+=v[k]*mult
 for r in lines(RAW/'val/VAL_WEIGHT_EXPOSURE.jsonl'):
  assert r['name'] not in seen;seen.add(r['name']);e=expect['per_input'][r['name']]
  for v in r['weight_exposure']:
   n=v['support'];cl=v['class'];assert v['G_weight_sum']==.5*n+v['G_high_pixels'] and v['S_weight_sum']==.5*n+v['S_high_pixels'] and 0<=v['S_high_pixels']<=n
   if v['basis'] in ['reference_valid','teacher_prediction_valid']:
    axis=0 if v['basis']=='reference_valid' else 1;cms={b:np.array(e['bands'][b]['confusion'],np.int64) for b in ['high','low']};counts={b:(cms[b].sum(1)[cl] if axis==0 else cms[b].sum(0)[cl]) for b in cms};assert n==counts['high']+counts['low'] and v['G_high_pixels']==counts['high']
   add('source_val535',v['basis'],'unique535',cl,v,roi=r['ROI'])
  for basis in ['reference_valid','teacher_prediction_valid','teacher_prediction_geometry']:
   vs=[v for v in r['weight_exposure'] if v['basis']==basis];assert len(vs)==3
   if basis=='teacher_prediction_geometry':assert sum(v['G_high_pixels'] for v in vs)==sum(v['S_high_pixels'] for v in vs)==131072 and sum(v['support'] for v in vs)==262144
 assert len(seen)==535;freq=collections.Counter(v['source_sample_id'] for v in js(RAW/'SOURCE_SEQUENCE_LOCK.json'));weights={v['sample_id']:v for v in js(RAW/'WEIGHT_MANIFEST.json')};seen=set()
 for r in lines(RAW/'replay/REPLAY_WEIGHTS.jsonl'):
  sid=r['sample_id'];assert sid not in seen;seen.add(sid);assert r['repeat_count']==freq[sid]
  for k in ['confidence_raw_SHA','G_SHA','S_SHA','permutation_SHA','stable_rank_SHA']:assert r['weight_proof'][k]==weights[sid][k]
  assert r['weight_proof']['numpy_version']=='1.26.4' and r['weight_proof']['local_PCG64_seed']==int.from_bytes(hashlib.sha256(('phase219|'+sid).encode()).digest()[:8],'big')
  vs=r['class_exposure'];assert sum(v['support'] for v in vs)==262144 and sum(v['G_high_pixels'] for v in vs)==sum(v['S_high_pixels'] for v in vs)==131072
  for v in vs:
   n=v['support'];assert v['G_weight_sum']==.5*n+v['G_high_pixels'] and v['S_weight_sum']==.5*n+v['S_high_pixels'];assert 0<=v['G_high_pixels']<=n and 0<=v['S_high_pixels']<=n
   for method,mult in [('unique256',1),('original4000_sequence',freq[sid])]:add('replay256','teacher_prediction_geometry',method,v['teacher_prediction_class'],v,mult)
 assert len(seen)==256 and sum(freq.values())==4000
 for r in csvrows(O/'TABLE_B_WEIGHT_ALLOCATION.csv')+csvrows(O/'VAL_WEIGHT_ALLOCATION_ALL_ROIS.csv'):
  key=r['pool'],r['basis'],r['repetition'],CS.index(r['class']);v=wb_roi[key+(r['ROI'],)] if 'ROI' in r else wb[key]
  for k in ['support','G_high_pixels','S_high_pixels','G_weight_sum','S_weight_sum']:assert float(r[k])==v[k]
  if v['support']:
   assert abs(float(r['G_mean_weight'])-v['G_weight_sum']/v['support'])<1e-12 and abs(float(r['S_mean_weight'])-v['S_weight_sum']/v['support'])<1e-12 and abs(float(r['high_confidence_fraction'])-v['G_high_pixels']/v['support'])<1e-12
 for s in [21701,21702]:
  for a in ['D','G','S']:assert complete['val']['student_prediction_SHA'][f'{s}/{a}']==expect['student_prediction_SHA'][f'{s}/{a}']
 assert complete['val']['teacher_prediction_SHA']==expect['teacher_prediction_SHA'];assert js(RAW/'val/GATE_ACCEPTED.json')['cloned_teacher_equal'];assert js(RAW/'REMOTE_COLLECTION_RECEIPT.json')['units']==4800
 cost=collections.Counter();ordinal=1
 for mode in ['val','replay']:
  for v in lines(RAW/mode/'CALL_LEDGER.jsonl'):assert v['ordinal']==ordinal;ordinal+=1;cost[v['kind']]+=1
 assert ordinal==4801 and sum(cost.values())==4800
 result={'status':'passed','raw_manifest_artifacts_SHA_checked':len(manifest),'explicit_six_axis_integer_cells_checked':allcells,'all535_images_two_seeds_107_ROIs':True,'all6_student_image_margins_and_teacher_correctness_reproduced':True,'teacher_confidence_bands_reproduced':True,'direct_correction_damage_and_error_flow_identities':True,'all256_replay_weight_SHA_original_numpy1264':True,'unique_and4000_sequence_weight_sums_checked':True,'historical_class_error_decomposition_reproduced':True,'exact_call_ledger':dict(cost),'new_bootstrap_training_fits_model_calls':0,'seconds':time.monotonic()-clock,'code_SHA':sha(Path(__file__))};(O/'INDEPENDENT_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
