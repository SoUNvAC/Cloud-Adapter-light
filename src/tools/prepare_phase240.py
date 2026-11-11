"""Verify the frozen six-arm specification and construct a small SHA deployment."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase240/phase240_20261030';F=ROOT/'src/research_plans/phase240_frozen_20261030';P=ROOT/'outputs/phase236/phase236_20261027'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 assert not O.exists(),'Existing run: inspect rather than overwrite';(O/'code').mkdir(parents=True)
 for line in (F/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split(maxsplit=1);assert sha(ROOT/n.strip().lstrip('*'))==h,n
 for z in js(F/'PRIOR_IDENTITIES.json'):
  p=ROOT/z['path'];assert p.stat().st_size==z['bytes'] and sha(p)==z['sha256'],z['path']
 for p in F.iterdir():shutil.copy2(p,O/p.name)
 assert js(F/'LOCK.json')['seed']==21701
 train=js(F/'TRAIN_INPUTS.json');evals=js(F/'EVAL_INPUTS.json');diag=js(F/'DIAGNOSTIC_INPUTS.json')
 assert train==js(ROOT/'src/research_plans/phase236_frozen_20261027/TRAIN_INPUTS.json')
 assert evals==js(ROOT/'src/research_plans/phase236_frozen_20261027/EVAL_INPUTS.json')
 for domain in train:
  assert len(diag[domain])==8 and len({v['scene'] for v in diag[domain]})==8
  assert all(v in train[domain] for v in diag[domain])
  for split in evals[domain]:assert not {v['scene'] for v in train[domain]} & {v['scene'] for v in evals[domain][split]}
 source={};crefs=[]
 for d in ['L8','SPARCS']:
  source[d]={}
  for split in ['val','test']:
   rows={}
   for line in (P/f'delivery/{d}_{split}_PER_INPUT.jsonl').read_text().splitlines():
    v=json.loads(line)
    if v['source']:
     if v['name'] in rows:assert rows[v['name']]['prediction_SHA']==v['prediction_SHA'] and rows[v['name']]['confusion']==v['confusion']
     rows[v['name']]=v
   assert len(rows)==len(evals[d][split]);source[d][split]=rows
  frozen=next(z for z in js(F/'C_BASELINE_VAL_SELECTION.json') if z['domain']==d);pool=frozen['candidates'];assert len(pool)==5
  chosen=min(pool,key=lambda c:(-c['pooled_mIoU'],0 if c['source'] else 1,c['step'] or 0));assert chosen==frozen['selected']
  for split in ['val','test']:
   cache=js(P/f'delivery/{d}_{split}_SUMMARY.json')
   for role,step in [('C_selected',chosen['step']),('C4000',4000)]:
    existing=[v for v in cache if v['seed']==21701 and v['lr']==1e-4 and v['step']==step and (v['strategy']=='E' or (split=='test' and v['strategy']=='endpoint'))]
    assert existing,'Missing C reference: do not run a new C';v=existing[0];assert all(z['confusion']==v['confusion'] for z in existing)
    crefs.append(dict(domain=d,split=split,role=role,selected=chosen,data=v))
 save('EXPECTED_SOURCE236.json',source);save('C_REFERENCES.json',crefs)
 for n in ['NATIVE_PLAN.json','L8_VAL_NATIVE_REFERENCES.json','EXPECTED_NATIVE234.json','TRAIN_BINDING235.json']:shutil.copy2(P/'delivery'/n,O/n)
 valid=js(P/'delivery/FILL_GATE_ACCEPTED.json');assert valid['status']=='passed';module=P/'delivery/code/phase236_valid_mask.py';assert sha(module)==valid['module_SHA'];shutil.copy2(module,O/'code'/module.name)
 save('VALID_MASK_REUSE.json',{'SHA':sha(module),'original_gate_passed':True,'receipt_SHA':sha(P/'delivery/FILL_GATE_ACCEPTED.json')})
 save('DEPENDENCY_LOCK.json',js(P/'SAVED_STATE_CPU_AUDIT.json')['dependencies_SHA'])
 save('CONFIG_LOCK.json',{p.relative_to(ROOT/'src').as_posix():sha(p) for p in [ROOT/'src/configs/protocol/phase64_target_parent_rgb.py',ROOT/'src/configs/protocol/phase64_source_parent_rgb.py']})
 for n in ['phase240_common.py','phase240_gradients.py','train_phase240.py','diagnose_phase240.py','evaluate_phase240.py','controller_phase240.py']:
  p=ROOT/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,O/'code'/n)
 save('LOCAL_PREFLIGHT.json',{'status':'passed','frozen_input_SHA_verified':True,'C_five_candidate_selection_recomputed':True,'C_selected_and_endpoint_counts_available':True,'valid_mask_original_gate_reused':True,'new_calls':0,'updates':0})
 save('BUNDLE_SHA.json',{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()})
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O))
 print(json.dumps({'status':'prepared','bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size}))
if __name__=='__main__':main()
