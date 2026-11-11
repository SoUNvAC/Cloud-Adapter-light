"""Derive four H fits from exact historical D kernels; only lambda changes."""
from pathlib import Path
import json,hashlib,shutil,ast,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase224/phase224_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();(O/'historical').mkdir();history={};targets={};jobs={};flat={}
 for phase in range(217,224):
  date='20261023' if phase in [217,218,219] else '20261024';p=R/f'outputs/phase{phase}/phase{phase}_{date}';m=js(p/'DELIVERY_MANIFEST.json');assert sha(p/'DELIVERY_MANIFEST.json')==js(p/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((p/n).stat().st_size==v['bytes'] and sha(p/n)==v['SHA'] for n,v in m.items());history[str(phase)]={'date':date,'manifest_SHA':sha(p/'DELIVERY_MANIFEST.json'),'files':len(m)}
 for target,phase,date in [('SPARCS',217,'20261023'),('L8',222,'20261024')]:
  old=R/f'outputs/phase{phase}/phase{phase}_{date}';raw=old/'delivery';hist=O/'historical'/target;hist.mkdir();(hist/'evaluation').mkdir()
  support=js(raw/('PHASE216_EVAL_INPUT_LOCK.json' if phase==217 else 'INPUT_SUPPORT_LOCK.json'));targets[target]={'phase':phase,'date':date,'support':support}
  for pool,rows in support.items():flat['source' if pool=='source' else target+'_'+pool]=rows
  for p in (raw/'evaluation').glob('*.json'):
   if p.name.startswith('Source__') or any(p.name.startswith(f'{s}_{a}_a') for s in [21701,21702] for a in ['C','D']) or p.name=='SOURCE_SELECTION_LOCK.json':shutil.copy2(p,hist/'evaluation'/p.name)
  for seed in [21701,21702]:
   job=f'{target}_seed{seed}';d=O/job;d.mkdir();source=raw/f'seed{seed}';jobs[job]={'target':target,'seed':seed,'historical_phase':phase,'historical_root':f'/home/scv/shared/phase{phase}_{date}','historical_gate_SHA':sha(source/'GATE_ACCEPTED.json'),'original_schedule_SHA':sha(source/'TARGET_SOURCE_SCHEDULE.json'),'historical_D_path':f'/home/scv/shared/phase{phase}_{date}/seed{seed}/heads/D_iter4000.pth','historical_D_SHA':js(source/'D_TRAIN_COMPLETE.json')['head_SHA']}
   for name in ['TARGET_SOURCE_SCHEDULE.json','SOURCE_POOL_REQUEST.json','SOURCE_IDENTITY_ACCEPTED.json']:shutil.copy2(source/name,d/name)
   shutil.copy2(source/'GATE_ACCEPTED.json',d/'HISTORICAL_GATE_ACCEPTED.json');shutil.copy2(source/'TRAINING_CONFIG_LOCK.json',d/'HISTORICAL_TRAINING_CONFIG_LOCK.json');cfg=js(source/'TRAINING_CONFIG_LOCK.json');assert cfg['KL']['lambda']==1;cfg['KL']['lambda']=4;(d/'TRAINING_CONFIG_LOCK.json').write_text(json.dumps(cfg,indent=2))
   t=(raw/f'code/train_phase{phase}_seed{seed}.py').read_text();t=t.replace(f'/home/scv/shared/phase{phase}_{date}','/home/scv/shared/phase224_20261024').replace(f"ROOT=PARENT/'seed{seed}'",f"ROOT=PARENT/'{job}'").replace('<60000','<120000').replace('<14400','<21600').replace("'calls_cap':60000","'calls_cap':120000").replace("'GPU_seconds_cap':14400","'GPU_seconds_cap':21600").replace("'planned_units':20005","'planned_units':16004").replace("'planned_training_updates':{'C':4000,'D':4000}","'planned_training_updates':{'H':4000}")
   t=t.replace("for arm in ['C','D']:","for arm in ['H']:").replace("arm=='D'","arm=='H'").replace('total=tl+k;','total=tl+4*k;').replace("'source_KL':float(k.detach()),","'source_KL':float(k.detach()),'source_KL_scaled4':float((4*k).detach()),'lambda':4,'finite_gradients':True,").replace("counts['target_full_segmentor']==8002","counts['target_full_segmentor']==4001").replace('sum(counts.values())==20005','sum(counts.values())==16004')
   start=t.index("        phase='gate';trainmode();");end=t.index('        endpoints={}',start)
   gate=f'''        phase='gate';trainmode();torch.manual_seed({seed});torch.cuda.manual_seed_all({seed});np.random.seed({seed});random.seed({seed})
        model.decode_head.load_state_dict(initial);a,loga=target(schedule[0]);kl=source_kl(schedule[0]);historical=js(ROOT/'HISTORICAL_GATE_ACCEPTED.json');assert loga==historical['target_loss_D'] and abs(float(kl.detach()))<1e-6
        (a+4*kl).backward();gradH=[n for n,p in model.named_parameters() if p.grad is not None];assert gradH and all(n.startswith('decode_head.') for n in gradH);assert all(p.grad is None for p in teacher.parameters());frozen_check()
        dump(ROOT/'GATE_ACCEPTED.json',{{'target_loss_H':loga,'historical_target_loss_D':historical['target_loss_D'],'initial_KL':float(kl.detach()),'grad_H':gradH,'trainable_each':1316516,'nonhead_digest':digest(frozen),'teacher_digest':teacher_digest,'initial_head_digest':digest(initial),'initial_head_file_SHA':sha(ROOT/'heads/initial.pth'),'gate_updates':0,'lambda':4,'counts':dict(counts)}})
        model.zero_grad(set_to_none=True);del a,kl;torch.cuda.empty_cache()
'''
   t=t[:start]+gate+t[end:]
   # Both domains deny all validation/test pixels while fitting, including symlink aliases.
   if 'eval_paths=' not in t:
    t=t.replace('    denied=[];access=collections.Counter()',"    denied=[];access=collections.Counter()\n    eval_paths={str(Path(v[k]).resolve()) for rows in js(PARENT/'INPUT_SUPPORT_LOCK.json').values() for v in rows for k in ['image_path','mask_path']}")
    t=t.replace("        if 'cloudsen12_high_l1c/ann_dir/train/' in p", "        if str(Path(p).resolve()) in eval_paths:raise RuntimeError('224 fitting cannot read validation/test pixels')\n        if 'cloudsen12_high_l1c/ann_dir/train/' in p")
   ast.parse(t);(O/f'code/train_{job}.py').write_text(t,encoding='utf-8')
 for name in ['SOURCE_SEQUENCE_LOCK.json','PHASE216_SOURCE_POOL_REQUEST.json','PHASE216_EXECUTION_CODE_SNAPSHOT.json','PHASE216_RUNTIME_CODE_LOCK.json']:shutil.copy2(R/'outputs/phase217/phase217_20261023'/name,O/name)
 shutil.copy2(R/'outputs/phase222/phase222_20261024/delivery/L8_TRAIN_INPUT_LOCK.json',O/'L8_TRAIN_INPUT_LOCK.json');shutil.copy2(R/'outputs/phase222/phase222_20261024/delivery/INPUT_PREFLIGHT_ACCEPTED.json',O/'L8_SUPPORT_ACCEPTED.json')
 (O/'INPUT_SUPPORT_LOCK.json').write_text(json.dumps(flat,indent=2));(O/'EXPERIMENT_LOCK.json').write_text(json.dumps({'targets':targets,'jobs':jobs,'history':history,'lambda':4,'T':1,'planned_units':100460,'training_units':64016,'evaluation_units':36444,'caps':{'units':120000,'GPU_seconds':21600,'CPU_GiB':16,'GPU_GiB':20,'new_bytes':4000000000,'host_free_bytes':4*1024**3},'only_hyperparameter_change':'source_KL_lambda1_to4','protocol_SHA':sha(R/'src/research_plans/PHASE224_STRONGER_UNIFORM_SOURCE_KL_20261024.md')},indent=2))
 for n in ['controller_phase224.py','check_phase224.py','evaluate_phase224.py','collect_phase224.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text());shutil.copy2(p,O/'code'/n)
 (O/'CODE_LOCK.json').write_text(json.dumps({p.name:sha(p) for p in (O/'code').glob('*.py')},indent=2))
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED not started. Four H(lambda4,T1)4000 fits from Source, original217/222 sequences, old C/D reused. Planned100460/120000 units,6hGPU/16GiBCPU/20GiBGPU/4GBnew/4GiBhost. No timer/push/download/next phase.\n');print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size,'planned_units':100460}))
if __name__=='__main__':main()
