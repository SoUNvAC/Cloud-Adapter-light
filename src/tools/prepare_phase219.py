"""Generate four paired fixed fits from the actual217 kernel, not a new training recipe."""
from pathlib import Path
import ast,json,hashlib,shutil,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase219/phase219_20261023';OLD=R/'outputs/phase217/phase217_20261023'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
SOURCE_FUNCTION=r'''        from phase219_weighting import decode,rawsha,verify_small_formula
        weight_manifest=js(PARENT/'weight_preparation/WEIGHT_MANIFEST.json');weight_byid={r['sample_id']:r for r in weight_manifest};packed={}
        for sid in weight_byid:
            f=PARENT/f'weight_preparation/masks/{sid}.npz';assert sha(f)==weight_byid[sid]['packed_masks_SHA']
            with np.load(f,allow_pickle=False) as z:packed[sid]={a:z[a].copy() for a in ['G','S']}
        source_details={}
        def source_kl(row,arm):
            path=SRC/'data/cloudsen12_high_l1c/img_dir/train'/row['source_png_name'];sid=row['source_sample_id'];x=torch.from_numpy(load_rgb_image(path,512)).cuda()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                count('source_shared_backbone')
                with torch.no_grad():
                    features=model.extract_feat(wrapper.normalize(x));s=semantic(teacher,features,'source_teacher_head');logp=F.log_softmax(s.float(),dim=1);p=logp.exp();c=p.max(1).values[0].cpu().numpy();assert rawsha(c)==weight_byid[sid]['confidence_raw_SHA']
                student=semantic(model.decode_head,features,'source_student_head');logq=F.log_softmax(student.float(),dim=1);pixel=(p*(logp-logq)).sum(1);w=torch.from_numpy(decode(packed[sid][arm])).cuda();assert not w.requires_grad and w.mean().item()==1
                weighted=(w*pixel).mean();plain=pixel.mean();high=pixel[w==1.5].mean();low=pixel[w==.5].mean();assert torch.isfinite(weighted) and weighted.detach().item()>=-1e-6
                source_details.clear();source_details.update({'source_KL_unweighted':float(plain.detach()),'source_KL_weighted':float(weighted.detach()),'high_weight_region_KL':float(high.detach()),'low_weight_region_KL':float(low.detach()),'weight_SHA':weight_byid[sid][arm+'_SHA'],'teacher_confidence_SHA':weight_byid[sid]['confidence_raw_SHA'],'weights_require_grad':False})
                return weighted
'''
GATE=r'''        dump(ROOT/'SMALL_FORMULA_CHECK.json',verify_small_formula(torch))
        phase='gate';trainmode();model.decode_head.load_state_dict(initial);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
        a,loga=target(schedule[0]);kg=source_kl(schedule[0],'G');(a+kg).backward();gradG=[n for n,p in model.named_parameters() if p.grad is not None];model.zero_grad(set_to_none=True)
        model.decode_head.load_state_dict(initial);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
        b,logb=target(schedule[0]);ks=source_kl(schedule[0],'S');(b+ks).backward();gradS=[n for n,p in model.named_parameters() if p.grad is not None]
        assert loga==logb and abs(float(kg.detach()))<1e-6 and abs(float(ks.detach()))<1e-6 and gradG and gradS and all(n.startswith('decode_head.') for n in gradG+gradS)
        assert all(p.grad is None for p in teacher.parameters());frozen_check()
        dump(ROOT/'GATE_ACCEPTED.json',{'target_loss_G':loga,'target_loss_S':logb,'initial_G_KL':float(kg.detach()),'initial_S_KL':float(ks.detach()),'grad_G':gradG,'grad_S':gradS,'trainable_each':1316516,'nonhead_digest':digest(frozen),'teacher_digest':teacher_digest,'initial_head_digest':digest(initial),'gate_updates':0,'counts':dict(counts),'weight_parameters':0})
        model.zero_grad(set_to_none=True);del a,b,kg,ks;torch.cuda.empty_cache()
'''
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();oldmanifest=js(OLD/'DELIVERY_MANIFEST.json');assert sha(OLD/'DELIVERY_MANIFEST.json')==js(OLD/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((OLD/n).stat().st_size==r['bytes'] and sha(OLD/n)==r['SHA'] for n,r in oldmanifest.items())
 for n in ['PHASE216_SOURCE_POOL_REQUEST.json','PHASE216_SOURCE_IDENTITY_ACCEPTED.json','PHASE216_EVAL_INPUT_LOCK.json','PHASE216_EXECUTION_CODE_SNAPSHOT.json','PHASE216_RUNTIME_CODE_LOCK.json','SOURCE_SEQUENCE_LOCK.json','SOURCE_CACHE_SHAS.json']:shutil.copyfile(OLD/'delivery'/n if (OLD/'delivery'/n).exists() else OLD/n,O/n)
 for seed in [21701,21702]:
  d=O/f'seed{seed}';d.mkdir();raw=OLD/f'delivery/seed{seed}'
  for n in ['TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json','SOURCE_POOL_REQUEST.json','SOURCE_IDENTITY_ACCEPTED.json']:shutil.copyfile(raw/n,d/n)
  t=(OLD/f'delivery/code/train_phase217_seed{seed}.py').read_text(encoding='utf-8').replace("Path('/home/scv/shared/phase217_20261023')","Path('/home/scv/shared/phase219_20261023')")
  t=t.replace("str(SRC/'tools')]","str(SRC/'tools'),str(PARENT/'code')]")
  t=t.replace('<60000','<80000').replace("'calls_cap':60000","'calls_cap':80000").replace('20005','32008').replace("'C'","'G'").replace("'D'","'S'")
  a=t.index('        def source_kl(row):');b=t.index("        phase='gate';",a);t=t[:a]+SOURCE_FUNCTION+t[b:]
  a=t.index("        phase='gate';");b=t.index('        endpoints={}',a);t=t[:a]+GATE.replace('SEED',str(seed))+t[b:]
  t=t.replace("k=source_kl(row) if arm=='S' else tl.new_zeros(())","k=source_kl(row,arm)").replace("'source_sample_id':row['source_sample_id'] if arm=='S' else None","'source_sample_id':row['source_sample_id'],**source_details,'gradient_finite':True")
  t=t.replace("all(counts[k]==4001","all(counts[k]==8002").replace("'source_training_uses':4000","'source_training_uses':8000").replace("'source_gate_uses':1","'source_gate_uses':2")
  t=t.replace("        phase='building';step=0;", "        assert sha(PARENT/'weight_preparation/WEIGHT_MANIFEST.json')==js(PARENT/'PRETRAIN_LOCK.json')['weight_manifest_SHA']\n        dump(ROOT/'TRAINING_CONFIG_EFFECTIVE.json',{'original217_config_SHA':sha(ROOT/'TRAINING_CONFIG_LOCK.json'),'weighting':js(PARENT/'PRETRAIN_LOCK.json'),'new_arms':['G','S'],'same_target_and_model_seed':"+str(seed)+"})\n        phase='building';step=0;")
  ast.parse(t);(O/f'code/train_phase219_seed{seed}.py').write_text(t,encoding='utf-8')
 code={}
 for n in ['phase219_weighting.py','precompute_phase219.py','controller_phase219.py','evaluate_phase219.py','check_phase219.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copyfile(p,O/'code'/n)
 for p in (O/'code').glob('*.py'):code[p.name]=sha(p)
 (O/'CODE_LOCK.json').write_text(json.dumps(code,indent=2),encoding='utf-8');(O/'LOCAL_PREFLIGHT.json').write_text(json.dumps({'old217_manifest_SHA':sha(OLD/'DELIVERY_MANIFEST.json'),'old217_artifacts_verified':len(oldmanifest),'expected_units':73946,'breakdown':{'weight_preparation':512,'four_training_updates':64000,'paired_gates':16,'candidate_eval':8880,'Source_parity':3,'extra_teacher_val_heads':535},'cap':80000,'model_fits':4,'old_C_D_refit':False},indent=2),encoding='utf-8')
 with zipfile.ZipFile(O/'bootstrap.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('Current delivery: prepared, not launched. Fixed219 G/S two original217 seeds, expected73946 units,80000 cap, four4000 fits. No C/D refit or timers.\n',encoding='utf-8');print(json.dumps({'prepared':str(O),'units':73946,'bootstrap_bytes':(O/'bootstrap.zip').stat().st_size}))
if __name__=='__main__':main()
