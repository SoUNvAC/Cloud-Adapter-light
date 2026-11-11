"""Prepare only the fixed234 manifests, historical identity counts and execution code."""
from pathlib import Path
import ast,hashlib,json,re,shutil,zipfile
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase234/phase234_20261027';T=REPO/'src/tools'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 assert not R.exists();F=REPO/'src/research_plans/phase234_frozen_20261027'
 for line in (F/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split(maxsplit=1);assert sha(REPO/n)==h,n
 L=js(F/'LOCK.json');pin=L['prior_identity_lock'];assert sha(REPO/pin['path'])==pin['sha256']
 for n,h in L['input_manifest_sources'].items():assert sha(REPO/n)==h,n
 E=js(F/'EVAL_INPUTS.json');assert [len(E[d][s]) for d,s in [('L8','val'),('L8','test'),('SPARCS','val'),('SPARCS','test')]]==[963,2167,10,10]
 R.mkdir(parents=True);(R/'code').mkdir();(R/'expected').mkdir();index={}
 for entry in js(F/'EXPECTED_OLD_IDENTITIES.json'):
  p=REPO/entry['path'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'];name=p.name.split('__')[0];key='Source' if name=='Source' else name.replace('_C_a','_');n=f"{entry['domain']}_{entry['split']}_{key}.json";shutil.copy2(p,R/'expected'/n);index[f"{entry['domain']}/{entry['split']}/{key}"]='expected/'+n
 assert len(index)==36;save(R/'EXPECTED_INDEX.json',index)
 for n in ['LOCK.json','EVAL_INPUTS.json','L8_VAL_NATIVE_REFERENCES.json']:shutil.copy2(F/n,R/n)
 old=REPO/'outputs/phase229/phase229_20261026'
 for n in ['MODEL_PATHS.json','RUNTIME_CODE_SHA.json']:shutil.copy2(old/n,R/n)
 ref=REPO/'outputs/phase317/phase317_20261026/remote_backup/REFERENCE_LOCK.json';v=js(ref);assert v['L8_all_PNG_reproduced_exactly'];axis=v['L8_axes'];assert axis in ['x_row_y_col','x_col_y_row']
 raw={g:{'path':f"/home/scv/shared/data/l8_biome_raw/l8biome/{next(v['biome'] for v in E['L8']['test'] if v['scene']==g)}/{g}/{g}_fixedmask.TIF"} for g in sorted({v['scene'] for v in E['L8']['test']})};save(R/'TEST_NATIVE_PLAN.json',{'verified_axes':axis,'axis_evidence_SHA':sha(ref),'only_fixed_raw_directory':True,'scenes':raw,'test_native_decoding_after_val_lock_only':True})
 for m in ['Source','L8_21701','L8_21702']:shutil.copy2(REPO/f'outputs/phase233/phase233_20261027/delivery/evaluation/{m}.json',R/f'expected/VALID233_{m}.json')
 shutil.copy2(REPO/'src/research_plans/PHASE234_CLEAN_GRID_VAL_SELECTION_OUTPUT_MIX.md',R/'PROTOCOL.md')
 for n in ['execute_phase234.py']:
  ast.parse((T/n).read_text());shutil.copy2(T/n,R/'code'/n)
 save(R/'PREPARE_ACCEPTED.json',{'status':'passed','all_frozen_identity_SHA':True,'old_expected_files':36,'test_expected_files_not_decoded_for_analysis':True,'planned_calls':28350,'cap':30000,'native_test_plan':'metadata only; exact fixed paths, bind after both val selection seal'})
 save(R/'BUNDLE_SHA.json',{p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file()})
 with zipfile.ZipFile(R/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in [*js(R/'BUNDLE_SHA.json'),'BUNDLE_SHA.json']:z.write(R/n,n)
 (R/'HANDOFF.md').write_text('Current delivery: PREPARED. Fixed234, no training/fitting.28350planned complete inputs, cap30000/7200GPUstage secs/CPU16GiB/GPU20GiB/new1GiB/CPUanalysis4h. Bothval pools evaluated first; global VAL_SELECTION_LOCK SHA before any test native decoding or test performance data. L8testrawpaths metadata only until lock; validatedaxis x_row_y_col exactPNG check eachpatch, no guesses. Full9weightmodels perinput, Sourceonce;Pmixonline noextra neuralcalls,no scores saved. AllOLD Widentity checks217/222,233VALID overlap checks. Newtestnativeidentity/scene/status/Fillcounts sealed. PerinputcountsJSONL+differences, allgridselectedtable and 3fixedpairedscenebootstrap10000seed23401, noselectinsidebootstrap. Knownhistoricaltestexposed/exploratory; L8Shadowsyes6/no10diagnostic. NoautoPush/cleanup/workdirsync/newphase.\n',encoding='utf-8')
 print(json.dumps({'bytes':(R/'bootstrap.zip').stat().st_size,'SHA':sha(R/'bootstrap.zip')}))
if __name__=='__main__':main()
