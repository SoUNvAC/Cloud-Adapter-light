"""Freeze local inputs and stage operators; never edit prior deliveries."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase236/phase236_20261027';F=ROOT/'src/research_plans/phase236_frozen_20261027'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 assert not list(O.rglob('*.*')) if O.exists() else True
 (O/'code').mkdir(parents=True,exist_ok=True)
 for line in (F/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split(maxsplit=1);assert sha(ROOT/n.strip().lstrip('*'))==h
 for v in js(F/'PRIOR_IDENTITIES.json'):
  p=ROOT/v['path'];assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],v['path']
 T=js(F/'TRAIN_INPUTS.json');E=js(F/'EVAL_INPUTS.json')
 for domain in T:
  groups={v['scene'] for v in T[domain]};assert len(groups)==(18 if domain=='L8' else 60)
  for split,rows in E[domain].items():assert not groups&{v['scene'] for v in rows},(domain,split)
 for p in F.glob('*'):shutil.copy2(p,O/p.name)
 for name,source in {'NATIVE_PLAN.json':'outputs/phase235/phase235_20261027/NATIVE_PLAN.json','L8_VAL_NATIVE_REFERENCES.json':'outputs/phase234/phase234_20261027/L8_VAL_NATIVE_REFERENCES.json','EXPECTED_NATIVE234.json':'outputs/phase234/phase234_20261027/delivery/NATIVE_REFERENCES.json','TRAIN_BINDING235.json':'outputs/phase235/phase235_20261027/delivery/TRAIN_NATIVE_BINDING.json'}.items():shutil.copy2(ROOT/source,O/name)
 for n in ['phase236_common.py','phase236_valid_mask.py','gate_phase236.py']:
  p=ROOT/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,O/'code'/n)
 (O/'BUNDLE_SHA.json').write_text(json.dumps({str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()},indent=2))
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('''# Current delivery
Phase236 prepared. Synthetic Mask2Former Fill gate first; no fit until FILL_GATE_ACCEPTED passes.
Local outputs/phase236/phase236_20261027; remote /home/scv/shared/phase236_20261027.
Gate source in code/; no installed-package edits. Frozen protocol src/research_plans/PHASE236_MATCHED_OPTIMIZATION_SELECTION.md.
8 C fits only: L8/SPARCS x21701/21702 x1e-4/3e-5,4000 steps, checkpoints500/1000/2000/4000.
Gate must prove all-valid original loss/gradient parity, invalid-node invariance including bilinear support and auxiliary layers.
85000 total calls/32000 updates/6GPUh/CPU16GiB/GPU20GiB/remote8GiB/formal2GiB. No automatic237, push, work_dirs transfer, downloads or cleanup.
''',encoding='utf-8')
 print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'gate_only':True}))
if __name__=='__main__':main()
