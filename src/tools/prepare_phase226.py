"""Exact217 kernels plus frozen scalar image weights; historical artifacts read-only."""
from pathlib import Path
import ast,json,hashlib,shutil,zipfile,collections,statistics
from phase226_weights import load
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase226/phase226_20261025';OLD=R/'outputs/phase217/phase217_20261023';FROZEN=R/'src/research_plans/phase226_frozen_20261025'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();(O/'historical/evaluation').mkdir(parents=True);hist={}
 for phase in range(217,226):
  date='20261023' if phase<220 else '20261024';p=R/f'outputs/phase{phase}/phase{phase}_{date}';m=js(p/'DELIVERY_MANIFEST.json');assert sha(p/'DELIVERY_MANIFEST.json')==js(p/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((p/n).stat().st_size==v['bytes'] and sha(p/n)==v['SHA'] for n,v in m.items());hist[str(phase)]={'date':date,'manifest_SHA':sha(p/'DELIVERY_MANIFEST.json'),'files':len(m)}
 shutil.copytree(FROZEN,O/'weights');wl=js(FROZEN/'WEIGHT_LOCK.json')
 for n,h in wl['inputs'].items():assert sha(R/n)==h,n
 seq=js(OLD/'SOURCE_SEQUENCE_LOCK.json');weights,proof=load(O/'weights',seq);(O/'WEIGHT_ACCEPTED_LOCAL.json').write_text(json.dumps(proof,indent=2));dependency={}
 for seed in [21701,21702]:
  root=O/f'seed{seed}';root.mkdir();old=OLD/f'delivery/seed{seed}';losses=[json.loads(v) for v in (old/'D_LOSSES.jsonl').read_text().splitlines()];sc=collections.defaultdict(list)
  for v in losses:sc[str(v['source_sample_id'])].append(v['source_KL'])
  for sid,w in weights.items():assert len(sc[sid])==int(w['repeat_count']) and abs(statistics.mean(sc[sid])-float(w[f'mean_online_KL_seed{seed}']))<1e-12
  dep=js(old/'D_TRAIN_COMPLETE.json');dependency[str(seed)]={'historical_D_training_units':sum(dep['counts_delta'].values()),'historical_D_train_seconds':dep['train_seconds'],'D_loss_log_SHA':sha(old/'D_LOSSES.jsonl'),'build_and_shared_gate_cost_attribution':'not independently attributable; original217 complete receipt retained'}
  for n in ['TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json','SOURCE_POOL_REQUEST.json','SOURCE_IDENTITY_ACCEPTED.json']:shutil.copy2(old/n,root/n)
  shutil.copy2(old/'GATE_ACCEPTED.json',root/'HISTORICAL_GATE_ACCEPTED.json');t=(OLD/f'delivery/code/train_phase217_seed{seed}.py').read_text().replace('phase217_20261023','phase226_20261025').replace('<60000','<85000').replace("'calls_cap':60000","'calls_cap':85000").replace("'planned_units':20005","'planned_units':32008").replace("'planned_training_updates':{'C':4000,'D':4000}","'planned_training_updates':{'R':4000,'Q':4000}")
  t=t.replace("for arm in ['C','D']:","for arm in ['R','Q']:").replace("k=source_kl(row) if arm=='D' else tl.new_zeros(())","k=source_kl(row);weight_csv=float(weight_byid[str(row['source_sample_id'])][arm+'_weight']);weight=torch.tensor(weight_csv,device=k.device,dtype=torch.float32,requires_grad=False);assert not weight.requires_grad").replace('total=tl+k;','weighted=weight*k;total=tl+weighted;').replace("'source_KL':float(k.detach()),","'source_KL':float(k.detach()),'weight_CSV':weight_csv,'weight_used':float(weight),'weighted_KL':float(weighted.detach()),'weight_requires_grad':False,'gradient_finite':True,").replace("row['source_sample_id'] if arm=='D' else None","row['source_sample_id']").replace('del tl,k,total,grads','del tl,k,total,grads,weighted,weight').replace('all(counts[k]==4001','all(counts[k]==8002').replace('sum(counts.values())==20005','sum(counts.values())==32008').replace("'source_training_uses':4000","'source_training_uses':8000").replace("'source_gate_uses':1","'source_gate_uses':2")
  marker="        phase='building';step=0;GPU_START=time.monotonic();status()"
  extra="""        sys.path.insert(0,str(PARENT/'code'))
        from phase226_weights import load,verify_formula
        weight_byid,weight_proof=load(PARENT/'weights',schedule)
        dump(ROOT/'TRAINING_CONFIG_EFFECTIVE.json',{'original217_config_SHA':sha(ROOT/'TRAINING_CONFIG_LOCK.json'),'weight_proof':weight_proof,'only_change':'one frozen CSV scalar times original uniform source KL','lambda':1,'T':1})
""";assert marker in t;t=t.replace(marker,extra+marker)
  start=t.index("        phase='gate';trainmode();");end=t.index('        endpoints={}',start)
  gate=f'''        dump(ROOT/'SMALL_FORMULA_CHECK.json',verify_formula(torch));phase='gate';gate_losses={{}};gate_kl={{}};gate_grads={{}};history=js(ROOT/'HISTORICAL_GATE_ACCEPTED.json')
        for gate_arm in ['R','Q']:
            trainmode();model.decode_head.load_state_dict(initial);torch.manual_seed({seed});torch.cuda.manual_seed_all({seed});np.random.seed({seed});random.seed({seed})
            a,loga=target(schedule[0]);k=source_kl(schedule[0]);w=torch.tensor(float(weight_byid[str(schedule[0]['source_sample_id'])][gate_arm+'_weight']),dtype=torch.float32,device=k.device,requires_grad=False);assert not w.requires_grad and abs(float(k))<1e-6 and loga==history['target_loss_D']
            (a+w*k).backward();grads=[n for n,p in model.named_parameters() if p.grad is not None];assert grads and all(n.startswith('decode_head.') for n in grads);assert all(p.grad is None for p in teacher.parameters());frozen_check();gate_losses[gate_arm]=loga;gate_kl[gate_arm]=float(k);gate_grads[gate_arm]=grads;model.zero_grad(set_to_none=True)
        assert gate_losses['R']==gate_losses['Q'];dump(ROOT/'GATE_ACCEPTED.json',{{'target_losses':gate_losses,'historical_target_loss_D':history['target_loss_D'],'initial_KL':gate_kl,'gradients':gate_grads,'trainable_each':1316516,'nonhead_digest':digest(frozen),'teacher_digest':teacher_digest,'initial_head_digest':digest(initial),'gate_updates':0,'counts':dict(counts),'weight_requires_grad':False}})
        model.zero_grad(set_to_none=True);del a,k,w,grads;torch.cuda.empty_cache()
''';t=t[:start]+gate+t[end:];ast.parse(t);(O/f'code/train_phase226_seed{seed}.py').write_text(t,encoding='utf-8')
 shutil.copy2(OLD/'PHASE216_EVAL_INPUT_LOCK.json',O/'INPUT_SUPPORT_LOCK.json')
 for n in ['SOURCE_SEQUENCE_LOCK.json','PHASE216_SOURCE_POOL_REQUEST.json','PHASE216_EXECUTION_CODE_SNAPSHOT.json','PHASE216_RUNTIME_CODE_LOCK.json']:shutil.copy2(OLD/n,O/n)
 for p in (OLD/'delivery/evaluation').glob('*.json'):
  if p.name.startswith('Source__') or any(p.name.startswith(f'{s}_{a}_a') for s in [21701,21702] for a in ['C','D']) or p.name=='SOURCE_SELECTION_LOCK.json':shutil.copy2(p,O/'historical/evaluation'/p.name)
 hroot=R/'outputs/phase224/phase224_20261024/delivery/evaluation'
 for s in [21701,21702]:
  for a in [.25,.5,.75,1]:
   for pool in ['source','val','test']:shutil.copy2(hroot/f'SPARCS_{s}_H_a{a}__{pool}.json',O/f'historical/evaluation/{s}_H_a{a}__{pool}.json')
 shutil.copy2(hroot/'SOURCE_SELECTION_LOCK.json',O/'historical/H_SOURCE_SELECTION_LOCK.json');shutil.copy2(OLD/'delivery/COMPLETE.json',O/'historical/PHASE217_COMPLETE_COST.json');(O/'HISTORICAL_DEPENDENCY_COST.json').write_text(json.dumps({'two_D_fits_required':dependency,'total_D_training_units':sum(v['historical_D_training_units'] for v in dependency.values()),'total_D_training_recorded_seconds':sum(v['historical_D_train_seconds'] for v in dependency.values()),'weight_preparation_neural_calls':0,'historical_class_exposure_cost':'reused Phase220, original receipt retained; not free model information','original_Source_training_cost':'unknown','not_a_single_training_run_method':True},indent=2))
 (O/'EXPERIMENT_LOCK.json').write_text(json.dumps({'history':hist,'weights_SHA':proof,'historical_D':{str(s):{'path':f'/home/scv/shared/phase217_20261023/seed{s}/heads/D_iter4000.pth','SHA':js(OLD/f'delivery/seed{s}/D_TRAIN_COMPLETE.json')['head_SHA']} for s in [21701,21702]},'planned_units':74508,'training_units':64016,'evaluation_units':10492,'cap_units':85000,'GPU_seconds_cap':14400,'only_change':'frozen scalar image weight on source uniform KL; lambda=T=1','protocol_SHA':sha(R/'src/research_plans/PHASE226_SPARCS_IMAGE_REPLAY_WEIGHTING_REPLICATION_20261025.md')},indent=2))
 for n in ['phase226_weights.py','controller_phase226.py','check_phase226.py','evaluate_phase226.py','collect_phase226.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text());shutil.copy2(p,O/'code'/n)
 (O/'CODE_LOCK.json').write_text(json.dumps({p.name:sha(p) for p in (O/'code').glob('*.py')},indent=2))
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED not launched. Only SPARCS R/Q two original217 seeds4000, frozen CSV scalar source weights; no ranking/resampling/new weight fitting. Planned74508/85000 calls,4hGPU/16GiBCPU/20GiBGPU/4GBnew/4GiBhost. No timers/push/download/next phase.\n');print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size,'planned_units':74508}))
if __name__=='__main__':main()
