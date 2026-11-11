"""Seal only original source-val and unlabeled replay metadata and historical margins."""
from pathlib import Path
import json,csv,hashlib,shutil,zipfile,ast,collections
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase220/phase220_20261024';P217=R/'outputs/phase217/phase217_20261023';P218=R/'outputs/phase218/phase218_20261023';P219=R/'outputs/phase219/phase219_20261023';SHARED='/home/scv/shared'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();historical={}
 for d in [P217,P218,P219]:
  m=js(d/'DELIVERY_MANIFEST.json');assert sha(d/'DELIVERY_MANIFEST.json')==js(d/'DELIVERY_COMPLETE.json')['manifest_SHA']
  assert all((d/n).stat().st_size==v['bytes'] and sha(d/n)==v['SHA'] for n,v in m.items());historical[d.name]={'manifest_SHA':sha(d/'DELIVERY_MANIFEST.json'),'all_artifacts_verified':len(m)}
 audit=R/'outputs/research_audits/phase219_class_damage_20261024';am=js(audit/'SHA256.json');assert all(sha(audit/n)==v for n,v in am.items());shutil.copyfile(audit/'SHA256.json',O/'ORIGINAL_CLASS_AUDIT_SHA.json');repeated=list(csv.DictReader((audit/'REPEATED_GT1_MIOU_DAMAGE.csv').open(encoding='utf-8-sig')));cohort={s:{v['ROI'] for v in repeated if int(v['seed'])==s} for s in [21701,21702]};assert cohort[21701]==cohort[21702] and len(cohort[21701])==13
 save('FOCUS_LOCK.json',{'historical_four':['ROI_0050','ROI_1013','ROI_1404','ROI_4813'],'both_seeds_G_minus_D_mIoU_below_minus1pp_13':sorted(cohort[21701]),'posthoc_no_selection_or_intervention':True,'original_audit_SHA':am})
 model={'Source':{'path':'/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth','SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'},'students':{}}
 for s in [21701,21702]:
  for a in ['D','G','S']:
   old=P217 if a=='D' else P219;v=js(old/f'delivery/seed{s}/{a}_TRAIN_COMPLETE.json');model['students'][f'{s}/{a}']={'path':f'{SHARED}/{old.name}/seed{s}/heads/{a}_iter4000.pth','SHA':v['head_SHA'],'steps':4000,'source_phase':217 if a=='D' else 219}
 save('MODEL_LOCK.json',model)
 val=js(P219/'delivery/evaluation/INPUT_SUPPORT_LOCK.json')['source'];official=list(csv.DictReader((R/'outputs/phase65/source_metadata/acquired/val_metadata.csv').open(encoding='utf-8-sig')));assert len(val)==len(official)==535
 support=[]
 for v in val:
  meta=official[int(Path(v['name']).stem)];assert meta['roi_id']==v['scene'];support.append({'name':v['name'],'sample_id':meta['index'],'ROI':v['scene'],'image_path':v['image_path'],'mask_path':v['mask_path'],'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256']})
 sequence=js(P219/'SOURCE_SEQUENCE_LOCK.json');freq=collections.Counter(v['source_sample_id'] for v in sequence);assert len(freq)==256 and sum(freq.values())==4000
 replay=[{'sample_id':v['sample_id'],'name':v['png_name'],'ROI':v['roi_id'],'image_path':'/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c/img_dir/train/'+v['png_name'],'image_SHA':v['png_sha256'],'repeat_count':freq[v['sample_id']]} for v in js(P219/'PHASE216_SOURCE_POOL_REQUEST.json')['selected']]
 save('SUPPORT_LOCK.json',{'val':support,'replay':replay,'val_official_metadata_SHA':sha(R/'outputs/phase65/source_metadata/acquired/val_metadata.csv'),'official_sample_id_is_index_not_png_ordinal':True});shutil.copyfile(P219/'SOURCE_SEQUENCE_LOCK.json',O/'SOURCE_SEQUENCE_LOCK.json')
 expected={'teacher_prediction_SHA':js(P219/'delivery/evaluation/Source__source.json')['prediction_SHA'],'student_prediction_SHA':{},'per_input':{v['name']:{'ROI':v['ROI'],'students':{},'bands':{}} for v in support}}
 for v in lines(P219/'delivery/evaluation/TEACHER_CONFIDENCE_REFERENCE.jsonl'):
  e=expected['per_input'][v['name']];e['high_mask_SHA']=v['high_mask_SHA'];e['bands'][v['band']]={'confusion':v['confusion'],'valid_pixels':v['valid_pixels'],'geometry_pixels':v['geometry_pixels'],'reference_class_support':v['reference_class_support'],'reference_class_errors':v['reference_class_errors']}
 for v in lines(P218/'delivery/PER_IMAGE.jsonl'):
  if v['pool']=='Source_val535' and v['arm']=='D':
   e=expected['per_input'][v['name']];e['teacher_prediction_SHA']=v['teacher_prediction_SHA'];e['students'][f"{v['seed']}/D"]={'confusion':[[sum(v['reference_teacher_student_counts'][i][t][j] for t in range(3)) for j in range(3)] for i in range(3)],'reference_teacher_student_counts':v['reference_teacher_student_counts'],'prediction_SHA':v['student_prediction_SHA']}
 for v in lines(P219/'delivery/evaluation/ENDPOINT_CORRECTNESS.jsonl'):
  e=expected['per_input'][v['name']];assert e['teacher_prediction_SHA']==v['teacher_prediction_SHA'];e['students'][f"{v['seed']}/{v['arm']}"]={'confusion':v['confusion'],'reference_teacher_student_counts':v['reference_teacher_student_counts']}
 for s in [21701,21702]:
  for a in ['D','G','S']:
   d=P217 if a=='D' else P219;raw=js(d/f'delivery/evaluation/{s}_{a}_a1__source.json');expected['student_prediction_SHA'][f'{s}/{a}']=raw['prediction_SHA']
   lookup={v['name']:v['confusion'] for v in raw['per_input']}
   for name,e in expected['per_input'].items():assert e['students'][f'{s}/{a}']['confusion']==lookup[name]
 assert all(len(e['students'])==6 and len(e['bands'])==2 for e in expected['per_input'].values());save('EXPECTED_VAL.json',expected)
 for n in ['PHASE216_EXECUTION_CODE_SNAPSHOT.json','PHASE216_RUNTIME_CODE_LOCK.json']:shutil.copyfile(P219/n,O/n)
 shutil.copyfile(P219/'delivery/weight_preparation/WEIGHT_MANIFEST.json',O/'WEIGHT_MANIFEST.json');shutil.copyfile(P219/'delivery/code/phase219_weighting.py',O/'code/phase219_weighting.py')
 for n in ['execute_phase220.py','controller_phase220.py','check_phase220.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copyfile(p,O/'code'/n)
 save('CODE_LOCK.json',{p.name:sha(p) for p in (O/'code').glob('*.py')});save('PREFLIGHT.json',{'historical_unchanged':historical,'expected_units':4800,'breakdown':{'source_val':4280,'first_input_gate_clone_and7_official_fulls':8,'replay':512},'cap':6000,'GPU_seconds_cap':3600,'new_training_fits_backwards_target_eval':0,'source_val_S_shuffle':'diagnostic only, same phase219|official val sample_id PCG64 rule, not replay training masks','label_boundary':'source-val diagnosis only; no replay train labels or official source test'})
 with zipfile.ZipFile(O/'bootstrap.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not launched. Frozen220 val535 six heads and replay256 no labels. Expected4800 units incl8 gate calls; cap6000/1hGPU. No training/target/thresholds/download/timer.\n',encoding='utf-8');print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size,'expected_units':4800}))
if __name__=='__main__':main()
