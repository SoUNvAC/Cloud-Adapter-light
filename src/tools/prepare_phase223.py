"""Lock eight old endpoint/interpolant identities and only original source-val inputs."""
from pathlib import Path
import json,hashlib,ast,shutil,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase223/phase223_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();(O/'expected').mkdir();models={};history={}
 for target,phase,date in [('SPARCS',217,'20261023'),('L8',222,'20261024')]:
  d=R/f'outputs/phase{phase}/phase{phase}_{date}';m=js(d/'DELIVERY_MANIFEST.json');assert sha(d/'DELIVERY_MANIFEST.json')==js(d/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((d/n).stat().st_size==v['bytes'] and sha(d/n)==v['SHA'] for n,v in m.items());history[target]={'phase':phase,'manifest_SHA':sha(d/'DELIVERY_MANIFEST.json'),'files':len(m)}
  for s in [21701,21702]:
   for a in ['C','D']:
    key=f'{target}_{s}_{a}';end=js(d/f'delivery/seed{s}/{a}_TRAIN_COMPLETE.json');ih=js(d/'delivery/evaluation/INTERPOLATED_HEAD_HASHES.json')[f'{s}_{a}_a0.25.pth'];models[key]={'target':target,'seed':s,'arm':a,'endpoint_path':f'/home/scv/shared/phase{phase}_{date}/seed{s}/heads/{a}_iter4000.pth','endpoint_SHA':end['head_SHA'],'interpolant_path':f'/home/scv/shared/phase{phase}_{date}/interpolated_heads/{s}_{a}_a0.25.pth','interpolant_SHA':ih,'alpha':.25};shutil.copy2(d/f'delivery/evaluation/{s}_{a}_a0.25__source.json',O/f'expected/{key}.json')
 p=R/'outputs/phase217/phase217_20261023/delivery';shutil.copy2(p/'evaluation/Source__source.json',O/'expected/Source.json');support=js(p/'PHASE216_EVAL_INPUT_LOCK.json')['source'];assert len(support)==535 and len({v['scene'] for v in support})==107
 runtime=js(p/'PHASE216_EXECUTION_CODE_SNAPSHOT.json')['source_runtime_code_SHA'];runtime.update(js(p/'PHASE216_RUNTIME_CODE_LOCK.json')['historical_code_SHA'])
 audit=R/'outputs/research_audits/cross_target_residual_damage_20261024';audits=js(audit/'SHA256.json');assert all(sha(audit/n)==h for n,h in audits.items());(O/'audit').mkdir()
 for n in ['CASE_SUMMARY.csv','ALL_ROI_SOURCE_DAMAGE.csv','L8_SELECTED_WORST_ROI_CROSS_TARGET.csv']:shutil.copy2(audit/n,O/'audit'/n)
 lock={'source_path':'/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth','Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','models':models,'support':support,'runtime_SHA':runtime,'expected_SHA':{p.name:sha(p) for p in (O/'expected').glob('*.json')},'history':history,'audit_SHA':audits,'fixed_coordinates':[[0,0],[0,511],[127,127],[255,255],[384,128],[511,511]],'focus_ROI':'ROI_0124','alpha':.25,'expected_units':5360,'main_units':5350,'engineering_extra_units':10,'cap_units':6000,'GPU_seconds_cap':3600,'negative_KL_tolerance':1e-6,'formula_CPU_GPU_tolerance':1e-6,'quantile_rule':'numpy linear p10/p50/p90 on exact float32 semantic gaps converted to float64; all selected pixels, no sampling','interpretation':'same-alpha diagnosis; SPARCS21701 selected alpha0.5 is not evaluated here','no_training_or_bootstrap':True}
 (O/'INPUT_MODEL_LOCK.json').write_text(json.dumps(lock,indent=2))
 for n in ['execute_phase223.py','check_phase223.py','collect_phase223.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text());shutil.copy2(p,O/'code'/n)
 (O/'CODE_LOCK.json').write_text(json.dumps({p.name:sha(p) for p in (O/'code').glob('*.py')},indent=2))
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for d in ['code','expected','audit']:
   for p in (O/d).iterdir():z.write(p,p.relative_to(O))
  for n in ['CODE_LOCK.json','INPUT_MODEL_LOCK.json']:z.write(O/n,n)
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not launched. Fixed alpha0.25 eight old SPARCS/L8 C/D heads, sourceval535/107 only. Expected5360 units including10 gate extras, cap6000/1hGPU/16GiBCPU/20GiBGPU/2GBnew/4GiBhost. Zero fits, backward, bootstrap or target/replay/test input access. No timer/push/new phase. Missing old head must be reported, never retrain.\n');print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size,'units':5360}))
if __name__=='__main__':main()
