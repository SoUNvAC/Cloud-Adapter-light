"""Prepare a validation-only package. Never open test pixels or run old evaluators."""
import ast, hashlib, json, re, shutil, zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2]
O=R/'outputs/phase229/phase229_20261026'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 assert not (O/'bootstrap.zip').exists();O.mkdir(parents=True,exist_ok=True);(O/'code').mkdir(exist_ok=True);(O/'expected').mkdir(exist_ok=True)
 lock=js(R/'src/research_plans/phase229_frozen_20261026/LOCK.json')
 for n,h in js(R/'src/research_plans/phase229_frozen_20261026/SHA256.json').items():assert sha(R/n)==h
 support={};training={};evidence={};forbidden=[];modelpaths={}
 for domain,phase,date in [('SPARCS',217,'20261023'),('L8',222,'20261024')]:
  old=R/f'outputs/phase{phase}/phase{phase}_{date}';raw=old/'delivery';manifest={k.replace('\\','/'):v for k,v in js(old/'DELIVERY_MANIFEST.json').items()}
  def accepted(p):
   key=str(p.relative_to(old)).replace('\\','/');entry=manifest[key];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['SHA'];evidence[str(p.relative_to(R))]=sha(p);return js(p)
  allsupport=accepted(raw/('PHASE216_EVAL_INPUT_LOCK.json' if domain=='SPARCS' else 'INPUT_SUPPORT_LOCK.json'))
  support[domain]=sorted(allsupport['val'],key=lambda v:(v['scene'],v['name']))
  forbidden.extend(v[k] for pool in ['test','source'] for v in allsupport[pool] for k in ['image_path','mask_path'])
  training[domain]={}
  for seed in [21701,21702]:
   seq=accepted(raw/f'seed{seed}/TARGET_SOURCE_SCHEDULE.json');training[domain][str(seed)]=sorted({re.search(r'LC\w{19}',v['image_path']).group(0) for v in seq})
   modelpaths[f'{domain}_{seed}']=f'/home/scv/shared/phase{phase}_{date}/seed{seed}/heads/C_iter4000.pth'
   comp=accepted(raw/f'seed{seed}/C_TRAIN_COMPLETE.json');assert comp['head_SHA']==next(v['endpoint_head_sha256'] for v in lock['heads'] if v['domain']==domain and v['seed']==seed)
  for label in ['Source','21701_C_a0.25','21702_C_a0.25']:
   p=raw/f'evaluation/{label}__val.json';v=accepted(p);save(O/'expected'/f'{domain}_{label}.json',v)
  assert len(support[domain])==lock['input_pool_contract'][domain+'_val']['inputs']
  assert len({v['scene'] for v in support[domain]})==lock['input_pool_contract'][domain+'_val']['scenes']
 overlap={}
 for domain,other in [('L8','SPARCS'),('SPARCS','L8')]:
  overlap[domain]={s:sorted({v['scene'] for v in support[domain]} & set(training[other][s])) for s in ['21701','21702']}
 assert not any(v for d in overlap.values() for v in d.values()),overlap
 code=js(R/'outputs/phase217/phase217_20261023/PHASE216_RUNTIME_CODE_LOCK.json')['historical_code_SHA']
 code.update(js(R/'outputs/phase217/phase217_20261023/PHASE216_EXECUTION_CODE_SNAPSHOT.json')['source_runtime_code_SHA'])
 local_runtime_differences={n:{'frozen_SHA':h,'local_SHA':sha(R/'src'/n)} for n,h in code.items() if sha(R/'src'/n)!=h}
 save(O/'LOCAL_RUNTIME_DIFFERENCES.json',{'differences':local_runtime_differences,'action':'Do not deploy local runtime files. Remote must match frozen runtime SHA before inference; otherwise stop.'})
 for n,h in lock['evaluation_code_sources'].items():assert sha(R/n)==h,n
 save(O/'LOCK.json',lock);save(O/'INPUT_SUPPORT.json',support);save(O/'FORBIDDEN_PATHS.json',sorted(set(forbidden)));save(O/'MODEL_PATHS.json',modelpaths);save(O/'RUNTIME_CODE_SHA.json',code)
 save(O/'PREFLIGHT_LOCAL.json',{'status':'passed','validation_vs_other_training_scene_overlap':overlap,'training_scene_sets':training,'spatial_overlap':'unknown','historical_access':'validation pools already used; exact cross-head history unknown; not independent confirmation','identity_evidence':evidence,'no_test_pixel_reads':True,'strategy':'Recompute complete fixed matrix; no historical output cache reused. First Source and both own C heads in each pool gated against existing confusion before cross directions.'})
 shutil.copy2(R/'src/research_plans/PHASE229_LANDSAT_CROSS_HEAD_VALIDATION.md',O/'PROTOCOL.md')
 for name in ['execute_phase229.py']:
  p=R/'src/tools'/name;ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,O/'code'/name)
 save(O/'BUNDLE_SHA.json',{str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()})
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,str(p.relative_to(O)).replace('\\','/'))
 (O/'HANDOFF.md').write_text('Current delivery: prepared, not launched. Validation-only 5x2 fixed alpha0.25. Full matrix4865 plus six parity forwards, no training. Cap5200 complete inputs and3600 GPU-stage seconds. No test data; bounded host16GiB/GPU20GiB/newfiles200MB, free disk4GiB.\n',encoding='utf-8')
 print(json.dumps({'bootstrap_bytes':(O/'bootstrap.zip').stat().st_size,'bootstrap_SHA':sha(O/'bootstrap.zip'),'overlap':overlap}))
if __name__=='__main__':main()
