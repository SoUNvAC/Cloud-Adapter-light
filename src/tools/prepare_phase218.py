"""Lock only217 endpoints/source support; no new image/label selection or model calls."""
from pathlib import Path
import hashlib,json,shutil,ast,zipfile
R=Path(__file__).resolve().parents[2];OLD=R/'outputs/phase217/phase217_20261023';O=R/'outputs/phase218/phase218_20261023'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert js(OLD/'DELIVERY_COMPLETE.json')['status']=='passed' and sha(OLD/'DELIVERY_MANIFEST.json')==js(OLD/'DELIVERY_COMPLETE.json')['manifest_SHA']
 assert not (O/'INPUT_MODEL_LOCK.json').exists() and not (O/'EXECUTION_STARTED.json').exists()
 O.mkdir(parents=True,exist_ok=True);(O/'code').mkdir(exist_ok=True);(O/'expected217').mkdir(exist_ok=True)
 manifest={k.replace('\\','/'):v for k,v in js(OLD/'DELIVERY_MANIFEST.json').items()};needed=['PHASE216_SOURCE_POOL_REQUEST.json','PHASE216_EVAL_INPUT_LOCK.json','PHASE216_EXECUTION_CODE_SNAPSHOT.json','PHASE216_RUNTIME_CODE_LOCK.json']+[f'seed{s}/{a}_TRAIN_COMPLETE.json' for s in [21701,21702] for a in ['C','D']]+['evaluation/Source__source.json']+[f'evaluation/{s}_{a}_a1__source.json' for s in [21701,21702] for a in ['C','D']]
 identities={}
 for n in needed:
  p=OLD/'delivery'/n;key='delivery/'+n;assert p.stat().st_size==manifest[key]['bytes'] and sha(p)==manifest[key]['SHA'];identities[n]=sha(p)
  if n.startswith('evaluation/'):shutil.copyfile(p,O/'expected217'/Path(n).name)
 request=js(OLD/'delivery/PHASE216_SOURCE_POOL_REQUEST.json');sel=request['selected'];assert len(sel)==256 and len({x['sample_id'] for x in sel})==256 and len({x['roi_id'] for x in sel})==249
 replay=[{'name':x['png_name'],'sample_id':x['sample_id'],'ROI':x['roi_id'],'image_path':'/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c/img_dir/train/'+x['png_name'],'image_SHA':x['png_sha256']} for x in sel]
 val=[{'name':x['name'],'ROI':x['scene'],'image_path':x['image_path'],'mask_path':x['mask_path'],'image_SHA':x['image_sha256'],'mask_SHA':x['mask_sha256']} for x in js(OLD/'delivery/PHASE216_EVAL_INPUT_LOCK.json')['source']];assert len(val)==535 and len({x['ROI'] for x in val})==107
 models={f'{s}_{a}':{'seed':s,'arm':a,'path':f'/home/scv/shared/phase217_20261023/seed{s}/heads/{a}_iter4000.pth','SHA':js(OLD/f'delivery/seed{s}/{a}_TRAIN_COMPLETE.json')['head_SHA']} for s in [21701,21702] for a in ['C','D']}
 runtime=js(OLD/'delivery/PHASE216_EXECUTION_CODE_SNAPSHOT.json')['source_runtime_code_SHA'];runtime.update(js(OLD/'delivery/PHASE216_RUNTIME_CODE_LOCK.json')['historical_code_SHA'])
 lock={'source_path':'/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth','Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','models':models,'pools':{'Replay256':replay,'Source_val535':val},'runtime_SHA':runtime,'expected_SHA':{p.name:sha(p) for p in (O/'expected217').iterdir()},'focus_ROIs':['ROI_0050','ROI_4813','ROI_1013','ROI_1404'],'KL':'FP32 sum p_Source*(logsoftmax(S_Source)-logsoftmax(S_student)); T=1; unweighted geometric pixels','prediction':'217 FP32 clamp-positive semantic / pixel sum, argmax','gate_inputs':[{'pool':p,'name':rs[0]['name']} for p,rs in [('Replay256',replay),('Source_val535',val)]],'fixed_coordinates':[[0,0],[0,511],[127,127],[255,255],[384,128],[511,511]],'negative_KL_tolerance':1e-6,'gate_semantic_equality':'bitwise with original wrapper; all output pixels','expected_units':4758,'main_units':4746,'gate_extra_units':12,'formal_gate_inputs_reused_without_extra_forward':True,'cap_units':6000,'GPU_seconds_cap':3600,'new_bytes_cap':2000000000,'prior217_manifest_SHA':sha(OLD/'DELIVERY_MANIFEST.json'),'inputs_verified':identities,'training_updates':0}
 (O/'INPUT_MODEL_LOCK.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
 audits={}
 for folder in ['replay_coverage_kl_20261023','distillation_damage_flow_20261023']:
  for p in (R/'outputs/research_audits'/folder).glob('*'):
   if p.is_file():audits[str(p.relative_to(R))]={'bytes':p.stat().st_size,'SHA':sha(p)}
 (O/'HISTORICAL_AUDIT_IDENTITIES.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
 code={}
 for n in ['execute_phase218.py','check_phase218.py','collect_phase218.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copyfile(p,O/'code'/n);code[n]=sha(p)
 (O/'CODE_LOCK.json').write_text(json.dumps(code,indent=2),encoding='utf-8')
 with zipfile.ZipFile(O/'bootstrap.zip','w',zipfile.ZIP_DEFLATED) as z:
  for d in ['code','expected217']:
   for p in (O/d).iterdir():z.write(p,p.relative_to(O))
  for n in ['INPUT_MODEL_LOCK.json','CODE_LOCK.json']:z.write(O/n,n)
 (O/'HANDOFF.md').write_text('Current delivery: prepared, not launched. Expected4758 units, cap6000; zero training. Only execute_phase218.py once after SHA/input/runtime preflight. Never restart or replace sealed outputs without checking PID/EXIT and keeping consumed ledger. No timers, pushes or next phase.\n',encoding='utf-8')
 print(json.dumps({'prepared':str(O),'expected_units':4758,'gate_extra':12,'bootstrap_bytes':(O/'bootstrap.zip').stat().st_size}))
if __name__=='__main__':main()
