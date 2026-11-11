"""Local-only sealing; never starts neural work."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase241/phase241_20261030';PREP=R.parent/'phase241_preparation_20261030';OLD=ROOT/'outputs/phase236/phase236_20261027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def main():
 assert not (R/'bootstrap.zip').exists();assert js(PREP/'LOCAL_ACCEPTANCE.json')['passed'] if 'passed' in js(PREP/'LOCAL_ACCEPTANCE.json') else js(PREP/'LOCAL_ACCEPTANCE.json')['status']=='passed'
 for name,h in js(PREP/'INPUT_SHA256.json').items():assert sha(ROOT/name)==h,name
 code=R/'code';code.mkdir(exist_ok=True)
 names=['phase241_common.py','phase241_cpu_common.py','phase241_rank_merge.cpp','phase241_ranking.py','infer_phase241.py','calibrate_phase241.py','analyze_phase241.py','validate_phase241.py','finalize_phase241.py','controller_phase241.py','test_phase241_cpu.py']
 for name in names:
  src=ROOT/'src/tools'/name
  if src.suffix=='.py':compile(src.read_text(encoding='utf-8'),str(src),'exec')
  shutil.copy2(src,code/name)
 shutil.copy2(PREP/'MODEL_IDENTITIES.json',R/'MODEL_IDENTITIES.json');shutil.copy2(PREP/'SCENE_SUPPORT.csv',R/'SUPPORT.csv');shutil.copy2(OLD/'EVAL_INPUTS.json',R/'EVAL_INPUTS.json');shutil.copy2(OLD/'TRAIN_INPUTS.json',R/'TRAIN_INPUTS.json')
 for name in ['NATIVE_PLAN.json','L8_VAL_NATIVE_REFERENCES.json','EXPECTED_NATIVE234.json']:shutil.copy2(OLD/'delivery'/name,R/name)
 for name in ['DEPENDENCY_LOCK.json','CONFIG_LOCK.json']:
  shutil.copy2(ROOT/'outputs/phase240/phase240_20261030'/name,R/name)
 heads=js(PREP/'MODEL_IDENTITIES.json')['C'];expected={}
 for domain in ('L8','SPARCS'):
  for split in ('val','test'):
   with (OLD/'delivery'/f'{domain}_{split}_PER_INPUT.jsonl').open(encoding='utf-8') as f:
    for line in f:
     v=json.loads(line);model=None
     if v['source']:model='Source'
     elif v['strategy']=='E' and any(h['domain']==domain and h['seed']==v['seed'] and h['learning_rate']==v['lr'] and h['step']==v['step'] for h in heads):model='C'+str(v['seed'])
     if model is None:continue
     rec={k:v[k] for k in ('domain','split','name','scene','confusion','prediction_SHA')};rec['model']=model;key=(domain,split,v['name'],model)
     if key in expected:assert expected[key]==rec
     expected[key]=rec
 assert len(expected)==9450,len(expected)
 (R/'EXPECTED236.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in expected.values()),encoding='utf-8')
 save(R/'LOCK.json',{'protocol_SHA':sha(ROOT/'src/research_plans/PHASE241_SOURCE_C_RANKING_ALIGNMENT_AUDIT.md'),'preparation_input_SHA':js(PREP/'INPUT_SHA256.json'),'models':js(PREP/'MODEL_IDENTITIES.json'),'limits':{'calls':9600,'training':0,'GPU_seconds':14400,'CPU_seconds':43200,'remote_bytes':60*1024**3,'minimum_start_free_bytes':80*1024**3,'reserve_bytes':10*1024**3},'score':'wrapper nonnegative 3 semantic FP32 scores / channel sum; no extra softmax','sequence':['val','calibrate_and_seal','test','analyze','independent_validate','finalize'],'sampling_seed':24101,'bootstrap_seed':24101,'test_history':'already exposed; exploratory','L8_no10':'nominal reference, unknown shadow completeness','remote_scheduled_tasks':0})
 files=[*['code/'+n for n in names],'MODEL_IDENTITIES.json','SUPPORT.csv','EVAL_INPUTS.json','TRAIN_INPUTS.json','NATIVE_PLAN.json','L8_VAL_NATIVE_REFERENCES.json','EXPECTED_NATIVE234.json','EXPECTED236.jsonl','DEPENDENCY_LOCK.json','CONFIG_LOCK.json','LOCK.json','REMOTE_PREFLIGHT.json']
 save(R/'BUNDLE_SHA.json',{n:sha(R/n) for n in files})
 with zipfile.ZipFile(R/'bootstrap.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in files+['BUNDLE_SHA.json']:z.write(R/name,name)
 save(R/'EXECUTION_LOCAL_ACCEPTANCE.json',{'passed':True,'syntax_files':len(names)-1,'expected_records':len(expected),'bundle_SHA':sha(R/'bootstrap.zip'),'bytes':(R/'bootstrap.zip').stat().st_size,'new_neural_calls':0})
 print(json.dumps(js(R/'EXECUTION_LOCAL_ACCEPTANCE.json')))
if __name__=='__main__':main()
