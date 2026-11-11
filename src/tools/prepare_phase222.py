"""Derive fixed L8 C/D kernels from sealed217, without changing historical files."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase222/phase222_20261024';P217=R/'outputs/phase217/phase217_20261023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();historical={}
 for phase,date in [(217,'20261023'),(219,'20261023'),(220,'20261024'),(221,'20261024')]:
  d=R/f'outputs/phase{phase}/phase{phase}_{date}';m=js(d/'DELIVERY_MANIFEST.json');assert sha(d/'DELIVERY_MANIFEST.json')==js(d/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((d/n).stat().st_size==v['bytes'] and sha(d/n)==v['SHA'] for n,v in m.items());historical[str(phase)]={'manifest_SHA':sha(d/'DELIVERY_MANIFEST.json'),'files':len(m)}
 for n in ['PHASE216_SOURCE_POOL_REQUEST.json','PHASE216_SOURCE_IDENTITY_ACCEPTED.json','PHASE216_RUNTIME_CODE_LOCK.json','PHASE216_EXECUTION_CODE_SNAPSHOT.json','SOURCE_SEQUENCE_LOCK.json']:
  shutil.copy2(P217/n,O/n)
 for seed in [21701,21702]:
  for kind in ['prepare','train']:
   p=P217/f'code/{kind}_phase217_seed{seed}.py'
   if not p.exists():p=P217/f'delivery/code/{kind}_phase217_seed{seed}.py'
   t=p.read_text(encoding='utf-8');t=t.replace('/home/scv/shared/phase217_20261023','/home/scv/shared/phase222_20261024').replace("PHASE64_TARGET='sparcs'","PHASE64_TARGET='l8'").replace('assert len(dataset)==60','assert len(dataset)==1560').replace("'target_labels':'all60 scenes dense labels both arms'","'target_labels':'original L8 Shadows=yes1560 patches; scenes separately locked'")
   t=t.replace('<60000','<120000').replace("'calls_cap':60000","'calls_cap':120000").replace('<14400','<21600').replace("'GPU_seconds_cap':14400","'GPU_seconds_cap':21600")
   if kind=='prepare' and seed==21701:
    start=t.index("        old_schedule=js(PARENT/'PHASE216_TARGET_SOURCE_SCHEDULE.json')")
    end=t.index('        schedule=[]',start);t=t[:start]+t[end:]
   # The stage-wide budget guard includes earlier completed children.
   t=t.replace('assert GPU_START is None or time.monotonic()-GPU_START<21600',"assert GPU_START is None or prior['GPU_seconds']+time.monotonic()-GPU_START<21600")
   t=t.replace('size<=4_000_000_000','size<=4_000_000_000')
   anchor='dataset=DATASETS.build(cfg.train_dataloader.dataset);dataset.full_init();assert len(dataset)==1560'
   t=t.replace(anchor,anchor+"\n        locked_train=js(PARENT/'L8_TRAIN_INPUT_LOCK.json');assert {(dataset.get_data_info(i)['img_path'],dataset.get_data_info(i)['seg_map_path']) for i in range(len(dataset))}=={(str(Path(v['image_path']).resolve()),str(Path(v['mask_path']).resolve())) for v in locked_train}")
   t=t.replace('    denied=[];access=collections.Counter()',"    denied=[];access=collections.Counter()\n    eval_paths={str(Path(v[k]).resolve()) for pool,rows in js(PARENT/'INPUT_SUPPORT_LOCK.json').items() for v in rows for k in ['image_path','mask_path']}\n")
   t=t.replace("        if 'cloudsen12_high_l1c/ann_dir/train/' in p", "        if str(Path(p).resolve()) in eval_paths:raise RuntimeError('Training/preparation cannot read validation/test pixels: '+p)\n        if 'cloudsen12_high_l1c/ann_dir/train/' in p")
   ast.parse(t);(O/f'code/{kind}_phase222_seed{seed}.py').write_text(t,encoding='utf-8')
 for source,new in [('outputs/phase212/phase212_retrain_20261020/evaluation/delivery/INPUT_SUPPORT_LOCK.json','OLD212_INPUT_SUPPORT.json'),('outputs/phase212/phase212_retrain_20261020/evaluation/delivery/Source__l8.json','EXPECTED_Source__val.json'),('outputs/phase213/phase213_20261021/delivery/l8_a0.json','EXPECTED_Source__test.json'),('outputs/phase217/phase217_20261023/delivery/evaluation/Source__source.json','EXPECTED_Source__source.json'),('outputs/phase213/phase213_20261021/TEST_LABEL_METADATA_AUDIT.json','OLD213_TEST_METADATA.json')]:
  p=R/source
  if not p.exists() and new=='EXPECTED_Source__source.json':p=P217/'delivery/evaluation/Source__source.json'
  shutil.copy2(p,O/new)
 for n in ['preflight_phase222.py','controller_phase222.py','check_phase222.py','evaluate_phase222.py','collect_phase222.py']:
  ast.parse((R/'src/tools'/n).read_text(encoding='utf-8'));shutil.copy2(R/'src/tools'/n,O/'code'/n)
 (O/'LOCAL_PREFLIGHT.json').write_text(json.dumps({'historical':historical,'protocol_SHA':sha(R/'src/research_plans/PHASE222_L8_UNIFORM_SOURCE_DISTILLATION_REPLICATION_20261024.md'),'planned_units':99190,'breakdown':{'training_and_gates':40010,'candidate_full':57570,'candidate_shared_D_head':1070,'diagnostic_teacher_head':535,'Source_parity_full':3,'shared_D_gate_full':2},'cap':120000,'GPU_seconds_cap':21600,'no_new_timer':True},indent=2))
 (O/'CODE_LOCK.json').write_text(json.dumps({p.name:sha(p) for p in (O/'code').glob('*.py')},indent=2))
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not started. Remote shared/phase222_20261024. First zero-call preflight validates L8 original inputs; then controller prepares both sequences and runs four C/D4000 fits, then fixed full/stratified evaluation. No timer, automatic restarts or push. Never substitute historical seed64 head.\n')
 print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'planned_units':99190}))
if __name__=='__main__':main()
