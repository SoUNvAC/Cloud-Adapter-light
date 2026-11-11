"""Validate calibration membership without opening validation score/label arrays."""
from pathlib import Path
import ast,hashlib,json,re,shutil,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase231/phase231_20261026'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();f=R/'src/research_plans/phase231_frozen_20261026';lock=js(f/'LOCK.json');ident={}
 for n,h in js(f/'SHA256.json').items():assert sha(R/n)==h;ident[n]=h
 for n,h in lock['identities'].items():assert sha(R/n)==h;ident[n]=h
 rows=js(f/'CALIBRATION_INPUTS.json');assert len(rows)==60 and len({v['scene'] for v in rows})==60 and rows==sorted(rows,key=lambda v:(v['scene'],v['name']))
 val=js(R/'src/research_plans/phase230_frozen_20261026/SPARCS_VAL_INPUTS.json');assert not ({v['scene'] for v in rows}&{v['scene'] for v in val});training={}
 for seed in [21701,21702]:
  seq=js(R/f'outputs/phase217/phase217_20261023/delivery/seed{seed}/TARGET_SOURCE_SCHEDULE.json');pairs={(Path(v['image_path']).name,Path(v['mask_path']).name) for v in seq};assert pairs=={(Path(v['image_path']).name,Path(v['mask_path']).name) for v in rows};training[str(seed)]={'unique_train_images':len(pairs),'all60_original_train_members':True}
 for n in ['LOCK.json','CALIBRATION_INPUTS.json']:shutil.copy2(f/n,O/n)
 old=R/'outputs/phase230/phase230_20261026';assert js(old/'DELIVERY_COMPLETE.json')['manifest_SHA']==sha(old/'DELIVERY_MANIFEST.json')
 for n in ['MODEL_PATHS.json','RUNTIME_CODE_SHA.json']:shutil.copy2(old/n,O/n)
 shutil.copy2(old/'delivery/MODEL_ACCEPTED.json',O/'EXPECTED_MODEL_ACCEPTED.json')
 save(O/'PREPARE_ACCEPTED.json',{'status':'passed','frozen_identities':ident,'membership':training,'validation_scene_overlap':[],'validation_arrays_opened':False,'calibration_requires_SP_label_supervision':True,'validation_metadata_only':'Frozen scene membership and score receipts identity, no validation NPY read before bias seal.','planned_calls':183,'calls_cap':220,'new_output_plan':'Only dfloat32 and uint8 labels cached; direct file stream, no durable TAR; remote+local expected~410MB<512MiB.'})
 shutil.copy2(R/'src/research_plans/PHASE231_TRAIN_ONLY_CLOUD_BIAS_CONTROL.md',O/'PROTOCOL.md')
 for n in ['execute_phase231.py','controller_phase231.py','fit_phase231.py','phase231_bias.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,O/'code'/n)
 save(O/'BUNDLE_SHA.json',{str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()})
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,str(p.relative_to(O)).replace('\\','/'))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not launched. Only60originalSPtrain calibration scenes,Source/L8seed21701/21702 alpha0.25.183planned calls(180+3repeat),cap220/1800GPUseconds. Fixed3 scalar biases bounds[-5,5],equal-scene BCE derivative60bisections,CPU8GiB/3600s. Before any val arrays read all3biases must be SHA sealed. No val neural forwards. dfloat32 cache+labeluint8 only; direct verified file-stream avoids TAR and stays512MiB combined. Then original230localval scores only,CPUmetrics/no gridsearch/bootstrap. No nextphase.\n',encoding='utf-8')
 print(json.dumps({'bytes':(O/'bootstrap.zip').stat().st_size,'SHA':sha(O/'bootstrap.zip')}))
if __name__=='__main__':main()
