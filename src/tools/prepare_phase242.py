"""Seal exact Phase241 cache references and fixed Phase242 CPU code locally."""
from pathlib import Path
import json,hashlib,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase242/phase242_20261031';P=ROOT/'outputs/phase241/phase241_20261030';D=P/'delivery'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def main():
 assert not (R/'bootstrap.zip').exists();assert js(P/'TRANSFER_ACCEPTED.json')['passed'];assert js(D/'INDEPENDENT_ACCEPTANCE.json')['passed']
 for n,h in js(D/'SHA256SUMS.json').items():assert sha(D/n)==h
 assert js(R/'REMOTE_PREFLIGHT.json')['free_bytes']>50*1024**3 and not js(R/'REMOTE_PREFLIGHT.json')['new_run_exists']
 code=R/'code';code.mkdir(exist_ok=True);names=['phase242_common.py','phase242_rank.cpp','analyze_phase242.py','test_phase242.py','smoke_phase242.py','validate_phase242.py','finalize_phase242.py','controller_phase242.py']
 for name in names:
  src=ROOT/'src/tools'/name
  if src.suffix=='.py':compile(src.read_text(encoding='utf-8'),str(src),'exec')
  shutil.copy2(src,code/name)
 index={}
 for split in ('val','test'):
  original=D/(split.upper()+'_SCORE_INDEX.jsonl');assert sha(original)==js(D/(split.upper()+'_COMPLETE.json'))['score_index_SHA'];rows=[json.loads(l) for l in original.read_text().splitlines()];index[split]=[v for v in rows if split!='test' or v['domain']!='L8' or v['usgs_shadows']=='yes']
  counts={}
  for v in index[split]:counts.setdefault((v['domain'],v['model']),set()).add(v['scene'])
  assert all(len(scenes)==(8 if d=='L8' and split=='val' else 6 if d=='L8' else 10) for (d,m),scenes in counts.items());assert len(counts)==6
 save(R/'INDEX_REFERENCES.json',index);shutil.copy2(D/'METRICS.json',R/'PRIOR_METRICS.json');shutil.copy2(D/'MODEL_IDENTITIES.json',R/'MODEL_IDENTITIES.json');th={d:{} for d in ('L8','SPARCS')}
 for v in js(D/'THRESHOLD_LOCK.json')['thresholds']:
  if v['state']=='raw':assert v['success'];th[v['domain']][v['model']]=v['threshold']
 assert all(len(v)==3 for v in th.values());save(R/'RAW_THRESHOLDS.json',th)
 metadata={d:{} for d in ('L8','SPARCS')}
 for d,pools in js(P/'EVAL_INPUTS.json').items():
  for split,rows in pools.items():
   for v in rows:metadata[d][v['scene']]={'original_biome':v.get('biome'),'biome':v.get('biome'),'usgs_shadows':v.get('usgs_shadows'),'condition_note':'metadata/posthoc reference; not proven physical cause'}
 save(R/'SCENE_METADATA.json',metadata);protocol=ROOT/'src/research_plans/PHASE242_SCENE_PAIR_AND_MATCHED_PIXEL_DIAGNOSIS.md';shutil.copy2(protocol,R/protocol.name)
 save(R/'LOCK.json',{'protocol_SHA':sha(protocol),'phase241_root':'/home/scv/shared/phase241_20261030','prior_files':{n:sha(D/n) for n in ['VAL_SCORE_INDEX.jsonl','TEST_SCORE_INDEX.jsonl','MODEL_IDENTITIES.json','METRICS.json','THRESHOLD_LOCK.json','INDEPENDENT_ACCEPTANCE.json']},'network_calls':0,'gradient_updates':0,'downloads':0,'new_thresholds':0,'new_bootstrap':0,'detector':'T_ij=median_all_valid(delta_i)-median_all_valid(delta_j); larger favorable; no absolute/inversion','g':'scene-equal median of val m_i, one per target/seed','J1':'per-scene all-valid median delta subtraction from Shadow log score; transductive diagnosis','top_six_rule':'test cross-scene pairs only; each target fixed three negative/three positive two-seed mean weighted deltas; ID ties; fewer if absent','limits':{'CPU_seconds':43200,'CPU_bytes':16*1024**3,'new_bytes':40*1024**3,'formal_bytes':500*1024**2,'reserve_bytes':10*1024**3,'parallel_workers_max':4},'read_only_Phase241':True,'L8_no10_excluded_before_cache_reads':True,'historical_test_exposure':True})
 files=['INDEX_REFERENCES.json','PRIOR_METRICS.json','MODEL_IDENTITIES.json','RAW_THRESHOLDS.json','SCENE_METADATA.json','LOCK.json','REMOTE_PREFLIGHT.json',protocol.name]+['code/'+n for n in names];save(R/'BUNDLE_SHA.json',{n:sha(R/n) for n in files})
 with zipfile.ZipFile(R/'bootstrap.zip','w',zipfile.ZIP_DEFLATED) as z:
  for n in files+['BUNDLE_SHA.json']:z.write(R/n,n)
 save(R/'LOCAL_ACCEPTANCE.json',{'passed':True,'syntax_python_files':len(names)-1,'prior_manifest_verified':True,'no10_filtered':True,'index_rows':{s:len(v) for s,v in index.items()},'bundle_SHA':sha(R/'bootstrap.zip'),'network_calls':0});print(json.dumps(js(R/'LOCAL_ACCEPTANCE.json')))
if __name__=='__main__':main()
