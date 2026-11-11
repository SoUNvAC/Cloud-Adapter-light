"""Original217/222 target kernels plus exact head sum L2-SP; no replay code."""
from pathlib import Path
import ast,json,hashlib,shutil,zipfile
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase227/phase227_20261025';TOP='/home/scv/shared/phase227_20261025'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();hist={};jobs={};support={};cost={}
 for phase in range(217,227):
  date='20261023' if phase<220 else '20261025' if phase==226 else '20261024';p=R/f'outputs/phase{phase}/phase{phase}_{date}';m=js(p/'DELIVERY_MANIFEST.json');assert sha(p/'DELIVERY_MANIFEST.json')==js(p/'DELIVERY_COMPLETE.json')['manifest_SHA'] and all((p/n).stat().st_size==v['bytes'] and sha(p/n)==v['SHA'] for n,v in m.items());hist[str(phase)]={'date':date,'manifest_SHA':sha(p/'DELIVERY_MANIFEST.json'),'files':len(m)}
 for domain,phase,date in [('SPARCS',217,'20261023'),('L8',222,'20261024')]:
  old=R/f'outputs/phase{phase}/phase{phase}_{date}';raw=old/'delivery';d=O/domain;(d/'historical/evaluation').mkdir(parents=True);evalsupport=js(raw/('PHASE216_EVAL_INPUT_LOCK.json' if domain=='SPARCS' else 'INPUT_SUPPORT_LOCK.json'));support[domain]=evalsupport;(d/'INPUT_SUPPORT_LOCK.json').write_text(json.dumps(evalsupport,indent=2));shutil.copy2(old/'SOURCE_SEQUENCE_LOCK.json',d/'SOURCE_SEQUENCE_LOCK.json')
  if domain=='L8':
   shutil.copy2(raw/'INPUT_PREFLIGHT_ACCEPTED.json',d/'INPUT_PREFLIGHT_ACCEPTED.json');shutil.copy2(raw/'L8_TRAIN_INPUT_LOCK.json',d/'L8_TRAIN_INPUT_LOCK.json')
  for p in (raw/'evaluation').glob('*.json'):
   if p.name.startswith('Source__') or any(p.name.startswith(f'{s}_{a}_a') for s in [21701,21702] for a in ['C','D']) or p.name=='SOURCE_SELECTION_LOCK.json':shutil.copy2(p,d/'historical/evaluation'/p.name)
  hroot=R/'outputs/phase224/phase224_20261024/delivery/evaluation'
  for s in [21701,21702]:
   for a in [.25,.5,.75,1]:
    for pool in ['source','val','test']:shutil.copy2(hroot/f'{domain}_{s}_H_a{a}__{pool}.json',d/f'historical/evaluation/{s}_H_a{a}__{pool}.json')
  shutil.copy2(hroot/'SOURCE_SELECTION_LOCK.json',d/'historical/H_SOURCE_SELECTION_LOCK.json');metad={}
  for seed in [21701,21702]:
   root=d/f'seed{seed}';root.mkdir();oldroot=raw/f'seed{seed}';job=f'{domain}/seed{seed}';cost[job]={}
   for arm,oldphase,oldpath in [('C',phase,oldroot),('D',phase,oldroot),('H',224,R/f'outputs/phase224/phase224_20261024/delivery/{domain}_seed{seed}')]:
    z=js(oldpath/f'{arm}_TRAIN_COMPLETE.json');cost[job][arm]={'phase':oldphase,'training_units':sum(z['counts_delta'].values()),'recorded_train_seconds':z['train_seconds'],'source_replay':arm!='C','head_SHA':z['head_SHA']}
   for n in ['TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json','SOURCE_IDENTITY_ACCEPTED.json']:shutil.copy2(oldroot/n,root/n)
   shutil.copy2(oldroot/'GATE_ACCEPTED.json',root/'HISTORICAL_GATE_ACCEPTED.json');metad[str(seed)]={'path':f'/home/scv/shared/phase{phase}_{date}/seed{seed}/heads/D_iter4000.pth','SHA':js(oldroot/'D_TRAIN_COMPLETE.json')['head_SHA']};jobs[job]={'domain':domain,'seed':seed,'old_phase':phase,'date':date,'schedule_SHA':sha(root/'TARGET_SOURCE_SCHEDULE.json'),'config_SHA':sha(root/'TRAINING_CONFIG_LOCK.json'),'entry':f'train_{domain}_{seed}.py'}
   t=(raw/f'code/train_phase{phase}_seed{seed}.py').read_text().replace(f'/home/scv/shared/phase{phase}_{date}',TOP+'/'+domain).replace('<60000','<125000').replace('<120000','<125000').replace('<14400','<21600').replace("'calls_cap':60000","'calls_cap':125000").replace("'calls_cap':120000","'calls_cap':125000").replace("'GPU_seconds_cap':14400","'GPU_seconds_cap':21600").replace("'planned_units':20005","'planned_units':8002").replace("'planned_training_updates':{'C':4000,'D':4000}","'planned_training_updates':{'W1':4000,'W10':4000}")
   # Remove every source replay reader and the teacher module, including dead code.
   a=t.index("        request=js(ROOT/'SOURCE_POOL_REQUEST.json')");b=t.index('        def tensor_sha',a);t=t[:a]+t[b:]
   a=t.index("        original_source_sequence=");b=t.index('        assert len(schedule)==4000',a);t=t[:a]+t[b:]
   a=t.index('        teacher=copy.deepcopy');b=t.index("        (ROOT/'heads').mkdir()",a)
   anchor="""        anchor={n:p.detach().float().clone() for n,p in model.decode_head.named_parameters()}
        assert sum(p.numel() for p in anchor.values())==1316516
        assert all(not a.requires_grad and a.data_ptr()!=dict(model.decode_head.named_parameters())[n].data_ptr() for n,a in anchor.items())
        anchor_digest=digest(anchor)
        dump(ROOT/'ANCHOR_PARAMETERS.json',{'numel':1316516,'anchor_digest':anchor_digest,'independent_storage':True,'parameters':[{'name':n,'shape':list(v.shape),'numel':v.numel(),'dtype':str(v.dtype)} for n,v in anchor.items()],'formula':'0.5*sum_all_head_elements_squared_difference_FP32','buffer_elements':0})
""";t=t[:a]+anchor+t[b:];t=t.replace('assert digest(teacher.state_dict())==teacher_digest','assert digest(anchor)==anchor_digest')
   a=t.index('        def semantic(');b=t.index("        phase='gate';",a)
   omega="""        def omega():
            value=.5*sum(((p.float()-anchor[n])**2).sum() for n,p in model.decode_head.named_parameters())
            assert value.dtype==torch.float32 and torch.isfinite(value) and float(value.detach())>=0
            return value
""";t=t[:a]+omega+t[b:]
   a=t.index("        phase='gate';");b=t.index('        endpoints={}',a)
   gate=f'''        from phase227_formula import check_formula
        dump(ROOT/'FORMULA_ACCEPTED.json',check_formula(torch,next(model.decode_head.parameters())))
        phase='gate';gate={{}};history=js(ROOT/'HISTORICAL_GATE_ACCEPTED.json')
        for name,gamma in [('W1',1),('W10',10)]:
            trainmode();model.decode_head.load_state_dict(initial);torch.manual_seed({seed});torch.cuda.manual_seed_all({seed});np.random.seed({seed});random.seed({seed});a,loga=target(schedule[0]);k=omega();assert float(k)==0 and loga==history['target_loss_C']
            ag=torch.autograd.grad(gamma*k,tuple(model.decode_head.parameters()),retain_graph=True);assert all(torch.count_nonzero(v)==0 for v in ag)
            (a+gamma*k).backward();grads=[n for n,p in model.named_parameters() if p.grad is not None];assert grads and all(n.startswith('decode_head.') for n in grads);frozen_check();gate[name]={{'target_loss':loga,'initial_Omega':float(k),'anchor_gradient_zero':True,'gradients':grads,'gamma':gamma}};model.zero_grad(set_to_none=True)
        dump(ROOT/'GATE_ACCEPTED.json',{{'arms':gate,'anchor_digest':anchor_digest,'trainable_each':1316516,'nonhead_digest':digest(frozen),'initial_head_digest':digest(initial),'gate_updates':0,'historical_target_loss_C':history['target_loss_C'],'counts':dict(counts)}});del a,k,ag,grads;torch.cuda.empty_cache()
''';t=t[:a]+gate+t[b:]
   t=t.replace("for arm in ['C','D']:","for arm,gamma in [('W1',1),('W10',10)]:").replace("k=source_kl(row) if arm=='D' else tl.new_zeros(())","k=omega();scaled=gamma*k").replace('total=tl+k;','total=tl+scaled;').replace("'source_KL':float(k.detach()),","'Omega':float(k.detach()),'gamma':gamma,'scaled_Omega':float(scaled.detach()),'L2_distance_before_update':float((2*k.detach()).sqrt()),'gradient_finite':True,").replace("'source_sample_id':row['source_sample_id'] if arm=='D' else None","'source_sample_id':None").replace('del tl,k,total,grads','del tl,k,scaled,total,grads').replace("'teacher_digest':teacher_digest","'anchor_digest':anchor_digest")
   t=t.replace("'train_seconds':time.monotonic()-trainclock,","'endpoint_Omega':float(omega().detach()),'endpoint_L2':float((2*omega().detach()).sqrt()),'gamma':gamma,'train_seconds':time.monotonic()-trainclock,")
   t=t.replace("assert counts['target_full_segmentor']==8002 and all(counts[k]==4001 for k in ['source_shared_backbone','source_teacher_head','source_student_head'])","assert dict(counts)=={'target_full_segmentor':8002}").replace('sum(counts.values())==20005','sum(counts.values())==8002').replace("'source_training_uses':4000,'source_gate_uses':1","'source_training_uses':0,'source_gate_uses':0,'source_train_RGB_reads':0,'source_teacher_forward':0,'source_replay_cache_reads':0")
   # Strong access guard on resolved paths; no source train pixels or any evaluation pixels while fitting.
   a=t.index('    denied=[];access=collections.Counter()');b=t.index('    sys.addaudithook(audit)',a)
   guard="""    denied=[];access=collections.Counter()
    eval_paths={str(Path(p).resolve()) for p in js(PARENT.parent/'FORBIDDEN_EVAL_PATHS.json')}
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(args[0])).replace('\\\\','/');resolved=str(Path(path).resolve()).replace('\\\\','/')
        if resolved in eval_paths or any(k in resolved for k in ['cloudsen12_high_l1c/img_dir/train/','cloudsen12_high_l1c/ann_dir/train/','cloudsen12_high_l1c/img_dir/test/','cloudsen12_high_l1c/ann_dir/test/']) or ('/evaluation/' in resolved and resolved.endswith(('.npy','.npz','.pth'))):
            denied.append(resolved);raise RuntimeError('227 training forbids source replay and evaluation pixels/caches')
""";t=t[:a]+guard+t[b:];t=t.replace("    os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')]","    os.chdir(SRC);sys.path[:0]=[str(PARENT.parent/'code'),str(SRC),str(SRC/'tools')]")
   assert 'source_kl' not in t and 'teacher=' not in t and 'load_rgb_image(path' not in t;ast.parse(t);(O/'code'/jobs[job]['entry']).write_text(t,encoding='utf-8')
  (d/'EXPERIMENT_LOCK.json').write_text(json.dumps({'historical_D':metad,'domain':domain},indent=2))
 for n in ['PHASE216_EXECUTION_CODE_SNAPSHOT.json','PHASE216_RUNTIME_CODE_LOCK.json']:shutil.copy2(R/'outputs/phase217/phase217_20261023'/n,O/n)
 (O/'FORBIDDEN_EVAL_PATHS.json').write_text(json.dumps(sorted({v[k] for domain in support.values() for rows in domain.values() for v in rows for k in ['image_path','mask_path']}),indent=2));(O/'HISTORICAL_TRAINING_COST.json').write_text(json.dumps(cost,indent=2));(O/'EXPERIMENT_LOCK.json').write_text(json.dumps({'jobs':jobs,'history':hist,'planned_units':102217,'training_units':32008,'evaluation_units':70209,'caps':{'units':125000,'GPU_seconds':21600,'CPU_GiB':16,'GPU_GiB':20,'new_bytes':4000000000,'host_free_bytes':4*1024**3},'gammas':[1,10],'anchored_parameters':1316516,'formula':'FP32 0.5 sum (theta-thetaSource)^2','source_training_replay':False,'protocol_SHA':sha(R/'src/research_plans/PHASE227_HEAD_L2SP_BASELINE_20261025.md')},indent=2))
 for n in ['phase227_formula.py','controller_phase227.py','check_phase227.py','evaluate_phase227.py','collect_phase227.py']:
  p=R/'src/tools'/n;ast.parse(p.read_text());shutil.copy2(p,O/'code'/n)
 (O/'CODE_LOCK.json').write_text(json.dumps({p.name:sha(p) for p in (O/'code').glob('*.py')},indent=2))
 with zipfile.ZipFile(O/'bootstrap.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.name!='bootstrap.zip':z.write(p,p.relative_to(O))
 (O/'HANDOFF.md').write_text('Current delivery: PREPARED, not launched. Eight W1/W10 fits4000, original217/222 target schedules, no source replay or teacher during training. Planned102217/125000 units,6hGPU. Shared teacher predictions only after all eight fits. No timer/push/download/nextphase.\n');print(json.dumps({'bootstrap_SHA':sha(O/'bootstrap.zip'),'bytes':(O/'bootstrap.zip').stat().st_size,'planned_units':102217}))
if __name__=='__main__':main()
