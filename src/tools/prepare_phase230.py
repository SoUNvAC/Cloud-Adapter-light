"""Freeze SPARCS validation only and prepare bounded score extraction."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase230/phase230_20261026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();(O/'expected').mkdir()
 f=R/'src/research_plans/phase230_frozen_20261026';lock=js(f/'LOCK.json');evidence={}
 for n,h in js(f/'SHA256.json').items():assert sha(R/n)==h;evidence[n]=h
 for n,h in lock['identities'].items():assert sha(R/n)==h;evidence[n]=h
 old=R/'outputs/phase229/phase229_20261026';assert js(old/'DELIVERY_COMPLETE.json')['manifest_SHA']==sha(old/'DELIVERY_MANIFEST.json')
 s=js(f/'SPARCS_VAL_INPUTS.json');assert s==js(old/'INPUT_SUPPORT.json')['SPARCS'] and len(s)==10 and len({x['scene'] for x in s})==10
 for name in lock['models']:shutil.copy2(old/f'delivery/evaluation/SPARCS_{name}.json',O/f'expected/{name}.json')
 for n in ['MODEL_PATHS.json','RUNTIME_CODE_SHA.json']:shutil.copy2(old/n,O/n)
 save(O/'LOCK.json',lock);save(O/'SPARCS_VAL_INPUTS.json',s);shutil.copy2(old/'delivery/MODEL_ACCEPTED.json',O/'EXPECTED_MODEL_ACCEPTED.json')
 save(O/'FREEZE_ACCEPTED.json',{'status':'passed','identities':evidence,'pool':'SPARCS original validation only','images':10,'expected_predictions':50,'planned_forwards':55,'repeats':'five fixed models on first sorted scene; no extra full-pool repeats','training_updates':0,'permission_note':'User explicitly authorized this execution chat to use existing SSH on2026-10-26; protocol no-SSH sentence referred to drafting chat.','score_operator':'float32 normalized semantic scores; d computed by NumPy log(max(float32 p,1e-12)), no second softmax','durable_budget_plan':'Remote score/labels~160MB plus local stream TAR~160MB and extracted cache~160MB; no remote durable TAR; combined<=512MiB.'})
 shutil.copy2(R/'src/research_plans/PHASE230_CLOUD_SURFACE_SCORE_DIAGNOSIS.md',O/'PROTOCOL.md')
 p=R/'src/tools/execute_phase230.py';ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,O/'code'/p.name)
 save(O/'BUNDLE_SHA.json',{str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()})
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,str(p.relative_to(O)).replace('\\','/'))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not launched. Only SPARCS10 validation inputs,5 models alpha0.25;50 main+5 first-scene repeats,cap80/1800sGPU.0training/fits/threshold search. Cache float32 normalized3class scores and10 uint8 reference labels in shared. Stream TAR directly to local without remote TAR, combined512MiB cap. CPU<=3600s,8GiB. No L8/source/test pixels, new images/dependencies/work_dirs sync/nextphase.\n',encoding='utf-8')
 print(json.dumps({'bytes':(O/'bootstrap.zip').stat().st_size,'SHA':sha(O/'bootstrap.zip')}))
if __name__=='__main__':main()
