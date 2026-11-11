"""Identity-only preparation, one global Source scalar, no target head weights."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile,collections
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase235/phase235_20261027';T=REPO/'src/tools'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 assert (R/'TRAIN_RAW_METADATA.json').exists() and not (R/'bootstrap.zip').exists();F=REPO/'src/research_plans/phase235_frozen_20261027'
 for line in (F/'SHA256SUMS.txt').read_text().splitlines():h,n=line.split(maxsplit=1);assert sha(REPO/n)==h,n
 for v in js(F/'PRIOR_IDENTITIES.json'):assert (REPO/v['path']).stat().st_size==v['bytes'] and sha(REPO/v['path'])==v['sha256'],v['path']
 train=js(F/'L8_TRAIN_INPUTS.json');E=js(F/'EVAL_INPUTS.json');assert len(train)==1560 and len({v['scene'] for v in train})==18 and all(v['new_split']=='target_train' for v in train);assert not ({v['scene'] for v in train}&{v['scene'] for split in E['L8'].values() for v in split});assert E==js(REPO/'outputs/phase234/phase234_20261027/EVAL_INPUTS.json')
 meta=js(R/'TRAIN_RAW_METADATA.json');assert all(v['exists'] for v in meta['scenes'].values()),'Missing fixed rawmask, stop L8 target';old=REPO/'outputs/phase234/phase234_20261027';oldmeta=js(old/'delivery/INPUT_IDENTITY_ACCEPTED.json')['test_raw_metadata_only'];assert oldmeta['verified_axes']=='x_row_y_col';(R/'code').mkdir()
 for n in ['LOCK.json','EVAL_INPUTS.json','L8_TRAIN_INPUTS.json','SPARCS_SOURCE_BIAS.json']:shutil.copy2(F/n,R/n)
 save(R/'NATIVE_PLAN.json',{'verified_axes':oldmeta['verified_axes'],'axis_evidence_SHA':sha(old/'delivery/INPUT_IDENTITY_ACCEPTED.json'),'train':meta['scenes'],'test':oldmeta['scenes']})
 shutil.copy2(old/'RUNTIME_CODE_SHA.json',R/'RUNTIME_CODE_SHA.json');shutil.copy2(old/'L8_VAL_NATIVE_REFERENCES.json',R/'L8_VAL_NATIVE_REFERENCES.json');shutil.copy2(old/'delivery/NATIVE_REFERENCES.json',R/'EXPECTED_NATIVE234.json')
 expected={}
 for domain,pools in E.items():
  expected[domain]={}
  for split in pools:
   rows=[]
   with (old/f'delivery/evaluation/{domain}_{split}.jsonl').open() as f:
    for line in f:
     v=json.loads(line)
     if v['seed']==21701 and v['strategy']=='W' and v['alpha']==0:rows.append(v)
   assert len(rows)==len(E[domain][split]);expected[domain][split]={v['name']:v for v in rows}
 save(R/'EXPECTED_RAW234.json',expected);b3={v['scene']:v for v in js(REPO/'outputs/phase232/phase232_20261027/VAL_PER_SCENE.json') if v['model']=='Source' and v['decision']=='B3'};assert len(b3)==10;save(R/'EXPECTED_B3_232.json',b3)
 coords=collections.Counter((v['scene'],v['x'],v['y']) for v in train);boxes=collections.defaultdict(list)
 for v in train:boxes[v['scene']].append((int(v['x']),int(v['y'])))
 overlap={g:sum(abs(x-a)<512 and abs(y-b)<512 for i,(x,y) in enumerate(vs) for a,b in vs[i+1:]) for g,vs in boxes.items()}
 save(R/'TRAIN_MEMBERSHIP_ACCEPTED.json',{'inputs':1560,'scenes':18,'scene_overlap_with_val_test':[],'new_split_not_directory_name':True,'duplicate_coordinate_entries':sum(n-1 for n in coords.values()),'spatially_overlapping_patch_pairs_per_scene':overlap,'policy':'keep original list, no dedup; scene equal and all listed valid pixels within scene equal','training_truth_before_bias_only':True})
 shutil.copy2(REPO/'src/research_plans/PHASE235_SOURCE_SINGLE_CLOUD_BIAS.md',R/'PROTOCOL.md')
 for n in ['phase235_common.py','execute_phase235.py','fit_phase235.py','controller_phase235.py','phase231_bias.py']:ast.parse((T/n).read_text());shutil.copy2(T/n,R/'code'/n)
 save(R/'PREPARE_ACCEPTED.json',{'status':'passed','all_prior_identities_verified':True,'new_scalar_fits':1,'SPARCS_reuse_only':True,'planned_calls':4710,'cap':4800,'temp_cache_not_returned':True})
 save(R/'BUNDLE_SHA.json',{p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file()})
 with zipfile.ZipFile(R/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in [*js(R/'BUNDLE_SHA.json'),'BUNDLE_SHA.json']:z.write(R/n,n)
 (R/'HANDOFF.md').write_text('Current delivery: PREPARED. Sourceonly235,18L8trainscenes1560patches nativefillbinding+1560forward,1sceneequal scalar root fixedbounds60steps/independentreplay; reuse232SPSourcebias1.5086972783056742(noSPtrainpixels). Sealboth beforeSourceval/test3150forwards.4710planned/4800total calls;GPUstagescumulative3600sec excludingCPUfit;CPUfit/analysis4h,16GiB/GPU20GiB/cache6GiB/formal512MiB. Tempd3/labelsvectors onlyshared/cache/,neverreturn;nativebindingshashes/receiptsretained. AllRaw234andSPvalB3_232exact. FrozenSource/2seed234W/P/CrefsCPUcounts only,notrerun. NoautoPush/workdirsync/cleanup/new236. Seeprotocol.\n',encoding='utf-8');print(json.dumps({'bootstrap_bytes':(R/'bootstrap.zip').stat().st_size,'SHA':sha(R/'bootstrap.zip')}))
if __name__=='__main__':main()
