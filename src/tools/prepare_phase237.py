"""Verify frozen rank subsets and five-candidate full60 references locally."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase237/phase237_20261028';F=ROOT/'src/research_plans/phase237_frozen_20261028';P=ROOT/'outputs/phase236/phase236_20261027'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def identity(v):return tuple(v[k] for k in ['seed','strategy','lr','step','alpha','source'])
def main():
 assert not list(O.rglob('*.*')) if O.exists() else True
 (O/'code').mkdir(parents=True,exist_ok=True)
 for line in (F/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split(maxsplit=1);assert sha(ROOT/n.strip())==h,n
 for v in js(F/'PRIOR_IDENTITIES.json'):p=ROOT/v['path'];assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],v['path']
 assert js(P/'DELIVERY_COMPLETE.json')['status']=='complete' and js(P/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
 original=js(ROOT/'src/research_plans/phase236_frozen_20261027/TRAIN_INPUTS.json')['SPARCS'];by={v['scene']:v for v in original};assert len(by)==60
 subsets=js(F/'SUBSET_INPUTS.json');rank=js(F/'DRAW_RANKINGS.json');overlap=js(F/'SUBSET_OVERLAP.json');evals=js(F/'EVAL_INPUTS.json');assert list(evals)==['val','test']
 for k in range(1,4):
  name=f'SUBSET_{k}';order=sorted(by,key=lambda g:(hashlib.sha256(f'phase237|subset{k}|{g}'.encode()).hexdigest(),g));rows=subsets[name];assert [v['scene'] for v in rows]==order[:18]
  assert [v['scene'] for v in rank[name]]==order
  for v in rows:assert v==by[v['scene']]
  for other in subsets:assert len({v['scene'] for v in rows}&{v['scene'] for v in subsets[other]})==overlap[name][other]
  for split,rows2 in evals.items():assert not {v['scene'] for v in rows}&{v['scene'] for v in rows2}
 candidates=js(P/'VAL_ALL_CANDIDATES.json');full=js(F/'FULL60_VAL_SELECTION.json');cachedtest=js(P/'delivery/SPARCS_test_SUMMARY.json');cacheval=js(P/'delivery/SPARCS_val_SUMMARY.json')
 def order(v):return (-v['pooled_mIoU'],0,0) if v['source'] else (-v['pooled_mIoU'],1,v['step'] if v['strategy']=='E' else v['alpha'])
 fullrefs=[]
 for v in full:
  pool=[r for r in candidates if (r['domain'],r['seed'],r['strategy'])==('SPARCS',v['seed'],v['strategy']) and r['lr'] in [None,1e-4]];assert len(pool)==5 and pool==v['candidates'];winner=min(pool,key=order);assert identity(winner)==identity(v['selected'])
  existing=[r for r in cachedtest if identity(r)==identity(winner)];assert len(existing)==1,'Missing selected full60 test: report gap, never new forward';fullrefs.append({'population':'FULL60','seed':v['seed'],'strategy':v['strategy'],'selected':winner,'val':next(r for r in cacheval if identity(r)==identity(winner)),'test':existing[0]})
 for seed in [21701,21702]:
  for split,cache in [('val',cacheval),('test',cachedtest)]:
   match=[r for r in cache if r['seed']==seed and r['lr']==1e-4 and r['step']==4000 and ((r['strategy']=='E' and split=='val') or (r['strategy']=='endpoint' and split=='test'))];assert len(match)==1;fullrefs.append({'population':'FULL60','seed':seed,'strategy':'endpoint','split':split,'data':match[0]})
 source={}
 for split in ['val','test']:
  rows={}
  for line in (P/f'delivery/SPARCS_{split}_PER_INPUT.jsonl').read_text().splitlines():
   v=json.loads(line)
   if v['source']:
    if v['name'] in rows:assert rows[v['name']]['prediction_SHA']==v['prediction_SHA'] and rows[v['name']]['confusion']==v['confusion']
    else:rows[v['name']]=v
  assert len(rows)==10;source[split]=rows
 for p in F.iterdir():shutil.copy2(p,O/p.name)
 save('FULL60_REFERENCES.json',fullrefs);save('EXPECTED_SOURCE236.json',source);save('FULL60_TRAIN_METADATA.json',original)
 save('DEPENDENCY_LOCK.json',js(P/'SAVED_STATE_CPU_AUDIT.json')['dependencies_SHA'])
 config={p.relative_to(ROOT/'src').as_posix():sha(p) for p in [ROOT/'src/configs/protocol/phase64_target_parent_rgb.py',ROOT/'src/configs/protocol/phase64_source_parent_rgb.py']};save('CONFIG_LOCK.json',config)
 save('LOCAL_PREFLIGHT.json',{'status':'passed','subset_rank_rule_recomputed':True,'all_subsets_equal_frozen':True,'overlap':overlap,'full60_selections_5_candidates_replayed':6,'full60_selected_test_available':True,'no_pixels_or_model_access':True,'subset_order_policy':'filter original236 dataset order to selected frozen membership, then unchanged InfiniteSampler seed','expected_fits':6,'updates_cap':24000,'calls_cap':26000})
 for n in ['phase237_common.py','train_phase237.py','evaluate_phase237.py','controller_phase237.py']:
  p=ROOT/'src/tools'/n;ast.parse(p.read_text());shutil.copy2(p,O/'code'/n)
 save('BUNDLE_SHA.json',{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()})
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O))
 print(json.dumps({'status':'prepared','bootstrap_SHA':sha(O/'bootstrap.zip'),'full60_new_calls':0}))
if __name__=='__main__':main()
