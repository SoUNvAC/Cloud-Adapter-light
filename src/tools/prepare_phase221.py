"""Generate N/P fits from actual frozen219 kernels; keep all historical artifacts read-only."""
from pathlib import Path
import ast,json,hashlib,shutil,zipfile,sys
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase221/phase221_20261024';OLD=R/'outputs/phase219/phase219_20261023';P220=R/'outputs/phase220/phase220_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
SOURCE_FUNCTION=r'''        from phase221_weighting import decode,rawsha,verify_small_formula
        weight_manifest=js(PARENT/'weight_preparation/WEIGHT_MANIFEST.json');weight_byid={r['sample_id']:r for r in weight_manifest};packed={}
        for sid in weight_byid:
            f=PARENT/f'weight_preparation/masks/{sid}.npz';assert sha(f)==weight_byid[sid]['packed_masks_SHA']
            with np.load(f,allow_pickle=False) as z:packed[sid]={a:z[a].copy() for a in ['G_bits','P_bits','teacher_prediction']}
        source_details={}
        def source_kl(row,arm):
            path=SRC/'data/cloudsen12_high_l1c/img_dir/train'/row['source_png_name'];sid=row['source_sample_id'];x=torch.from_numpy(load_rgb_image(path,512)).cuda()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                count('source_shared_backbone')
                with torch.no_grad():
                    features=model.extract_feat(wrapper.normalize(x));s=semantic(teacher,features,'source_teacher_head');logp=F.log_softmax(s.float(),dim=1);p=logp.exp();c=p.max(1).values[0].cpu().numpy();assert rawsha(c)==weight_byid[sid]['confidence_raw_SHA'];zz=s.clamp_min(0);tp=(zz/zz.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8);assert rawsha(tp)==weight_byid[sid]['teacher_prediction_SHA']
                student=semantic(model.decode_head,features,'source_student_head');logq=F.log_softmax(student.float(),dim=1);pixel=(p*(logp-logq)).sum(1);decoded=decode(packed[sid],weight_byid[sid],arm);assert rawsha(decoded)==weight_byid[sid][arm+'_SHA'];w=torch.from_numpy(decoded).cuda();labels=torch.from_numpy(tp).cuda()[None];assert not w.requires_grad and abs(w.mean().item()-1)<=1e-6
                weighted=(w*pixel).mean();plain=pixel.mean();assert torch.isfinite(weighted) and weighted.detach().item()>=-1e-6
                classes=[]
                for cl in range(3):
                    keep=labels==cl;n=int(keep.sum());classes.append({'class':cl,'support':n,'weight_mean':float(w[keep].mean()) if n else None,'unweighted_KL_mean':float(pixel[keep].mean().detach()) if n else None,'weighted_KL_contribution':float((w[keep]*pixel[keep]).sum().detach()/pixel.numel()) if n else 0.0})
                assert abs(sum(v['weighted_KL_contribution'] for v in classes)-float(weighted.detach()))<=2e-6 and all(v['weight_mean'] is None or abs(v['weight_mean']-1)<=1e-6 for v in classes)
                source_details.clear();source_details.update({'source_KL_unweighted':float(plain.detach()),'source_KL_weighted':float(weighted.detach()),'teacher_class_KL':classes,'weight_SHA':weight_byid[sid][arm+'_SHA'],'teacher_confidence_SHA':weight_byid[sid]['confidence_raw_SHA'],'teacher_prediction_SHA':weight_byid[sid]['teacher_prediction_SHA'],'weights_require_grad':False})
                return weighted
'''
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();historical={}
 for phase,d in [(217,R/'outputs/phase217/phase217_20261023'),(218,R/'outputs/phase218/phase218_20261023'),(219,OLD),(220,P220)]:
  m=js(d/'DELIVERY_MANIFEST.json');assert sha(d/'DELIVERY_MANIFEST.json')==js(d/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((d/n).stat().st_size==v['bytes'] and sha(d/n)==v['SHA'] for n,v in m.items());historical[str(phase)]={'manifest_SHA':sha(d/'DELIVERY_MANIFEST.json'),'all_verified':len(m)}
 for n in ['PHASE216_SOURCE_POOL_REQUEST.json','PHASE216_SOURCE_IDENTITY_ACCEPTED.json','PHASE216_EVAL_INPUT_LOCK.json','PHASE216_EXECUTION_CODE_SNAPSHOT.json','PHASE216_RUNTIME_CODE_LOCK.json','SOURCE_SEQUENCE_LOCK.json','SOURCE_CACHE_SHAS.json']:shutil.copyfile(OLD/n,O/n)
 save('SOURCE_VAL_SAMPLE_IDS.json',{v['name']:v['sample_id'] for v in js(P220/'delivery/SUPPORT_LOCK.json')['val']})
 for n,new in [('MODEL_LOCK.json','HISTORICAL_MODEL_LOCK.json'),('EXPECTED_VAL.json','EXPECTED_VAL.json'),('FOCUS_LOCK.json','FOCUS_LOCK.json')]:shutil.copyfile(P220/'delivery'/n,O/new)
 for seed in [21701,21702]:
  d=O/f'seed{seed}';d.mkdir();raw=OLD/f'delivery/seed{seed}'
  for n in ['TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json','SOURCE_POOL_REQUEST.json','SOURCE_IDENTITY_ACCEPTED.json']:shutil.copyfile(raw/n,d/n)
  t=(OLD/f'delivery/code/train_phase219_seed{seed}.py').read_text(encoding='utf-8').replace('phase219_20261023','phase221_20261024').replace('80000','85000').replace("'G'","'N'").replace("'S'","'P'")
  start=t.index('        from phase219_weighting import');end=t.index("        dump(ROOT/'SMALL_FORMULA_CHECK.json'",start);t=t[:start]+SOURCE_FUNCTION+t[end:]
  for a,b in [('target_loss_G','target_loss_N'),('target_loss_S','target_loss_P'),('initial_G_KL','initial_N_KL'),('initial_S_KL','initial_P_KL'),('grad_G','grad_N'),('grad_S','grad_P')]:t=t.replace(a,b)
  t=t.replace('source_kl(schedule[0],\'G\')','source_kl(schedule[0],\'N\')').replace('source_kl(schedule[0],\'S\')','source_kl(schedule[0],\'P\')')
  t=t.replace("sys.path[:0]=[str(SRC),str(SRC/'tools'),str(PARENT/'code')]","sys.path[:0]=[str(PARENT/'code'),str(SRC),str(SRC/'tools')]")
  ast.parse(t);(O/f'code/train_phase221_seed{seed}.py').write_text(t,encoding='utf-8')
 shutil.copyfile(OLD/'delivery/code/phase219_weighting.py',O/'code/phase219_weighting.py')
 for n in ['phase221_weighting.py','precompute_phase221.py','controller_phase221.py','evaluate_phase221.py','check_phase221.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copyfile(p,O/'code'/n)
 sys.path.insert(0,str(R/'src/tools'));from phase221_weighting import boundary_tests;from phase219_weighting import build
 import numpy as np
 before=np.random.get_state();g,_,_=build(np.ones((512,512),np.float32),'equal_confidence_fixture');assert (g[:131072]==1.5).all() and (g[131072:]==.5).all();after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:];save('PURE_ARRAY_BOUNDARY_TESTS.json',boundary_tests())
 save('CODE_LOCK.json',{p.name:sha(p) for p in (O/'code').glob('*.py')});save('LOCAL_PREFLIGHT.json',{'historical_unchanged':historical,'expected_units':75018,'breakdown':{'weight_preparation':512,'four_training_updates':64000,'paired_gates':16,'candidates_full_and_shared':8880,'extra_D_heads':1070,'extra_teacher_heads':535,'Source_parity':3,'P_shared_formal_path_gates':2},'cap':85000,'new_fits':4,'original_C_D_G_S_refit':False,'source_val_joint_axes':['high_low','reference','teacher','D','N','P']})
 with zipfile.ZipFile(O/'bootstrap.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not launched. Four N/P fixed4000 fits, expected75018/85000 units, old217-220 read-only. No timer/automatic push.\n',encoding='utf-8');print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size,'units':75018}))
if __name__=='__main__':main()
