"""Verify immutable216 delivery; derive unchanged training kernels for two seeds."""
from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'outputs/phase216/phase216_20261023';OUT=ROOT/'outputs/phase217/phase217_20261023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'code').mkdir()
    manifest=js(OLD/'DELIVERY_MANIFEST.json');assert sha(OLD/'DELIVERY_MANIFEST.json')==js(OLD/'DELIVERY_COMPLETE.json')['manifest_SHA']
    assert all((OLD/n).stat().st_size==r['bytes'] and sha(OLD/n)==r['sha256'] for n,r in manifest.items())
    original=ROOT/'src/tools/execute_phase216.py';assert sha(original)=='284ec89365f5dfccbd6082bc4e5ea232daaf539341b2d4e408023fa0856a1458'
    for n in ['SOURCE_POOL_REQUEST.json','SOURCE_IDENTITY_ACCEPTED.json','TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json','EXECUTION_CODE_SNAPSHOT.json','RUNTIME_CODE_LOCK.json','EVAL_INPUT_LOCK.json','COMPLETE.json']:
        shutil.copy2(OLD/'delivery'/n,OUT/('PHASE216_'+n))
    sequence=[{'source_sample_id':r['source_sample_id'],'source_png_name':r['source_png_name']} for r in js(OLD/'delivery/TARGET_SOURCE_SCHEDULE.json')]
    (OUT/'SOURCE_SEQUENCE_LOCK.json').write_text(json.dumps(sequence,indent=2),encoding='utf-8')
    text=original.read_text(encoding='utf-8').split("        active_arm=None;phase='evaluation_preflight'",1)[0]
    text=text.replace("ROOT=Path('/home/scv/shared/phase216_20261023')","PARENT=Path('/home/scv/shared/phase217_20261023')\nROOT=PARENT/'REPLACE_ROOT'")
    text=text.replace("assert not (ROOT/'EXECUTION_STARTED.json').exists()","prior=js(PARENT/'BUDGET_STATE.json')\n    assert not (ROOT/'EXECUTION_STARTED.json').exists()")
    text=text.replace("sum(counts.values())<30000","prior['units']+sum(counts.values())<60000")
    text=text.replace("time.monotonic()-GPU_START<14400","prior['GPU_seconds']+time.monotonic()-GPU_START<14400")
    text=text.replace("'planned_units':25000","'planned_units':20005").replace("'calls_cap':30000","'calls_cap':60000")
    text=text.replace("'common_Source_evaluation':'555 fresh calls, compare historical confusion; no uncertain cache reuse'","'evaluation_here':False")
    text=text.replace("return data,proof","return data,json.loads(json.dumps(proof,default=str))")
    text=text.replace("for p in ROOT.rglob('*')","for p in PARENT.rglob('*')")
    # Existing fixed recipe is reused exactly; only target RNG seeds change.
    suffix='''
        done.set();phase='complete';active_arm=None;step=0
        assert counts['target_full_segmentor']==8002 and all(counts[k]==4001 for k in ['source_shared_backbone','source_teacher_head','source_student_head'])
        assert sum(counts.values())==20005
        frozen_check()
        dump(ROOT/'ACCESS_AUDIT.json',{'forbidden_attempts':denied,'source_train_label_reads':0,'source_test_label_reads':0,'source_val_label_opens':access['source_val_label_file_opens'],'source_training_uses':4000,'source_gate_uses':1})
        dump(ROOT/'TRAINING_COMPLETE.json',{'counts':dict(counts),'total_units':sum(counts.values()),'GPU_seconds':time.monotonic()-GPU_START,'Source_SHA':sha(SOURCE),'head_SHA':{p.name:sha(p) for p in (ROOT/'heads').glob('*.pth')},'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'nonhead_digest':digest(frozen),'teacher_digest':teacher_digest,'seed':SEED_VALUE})
        status();(ROOT/'EXIT_CODE').write_text('0\\n')
    except BaseException:
        done.set();(ROOT/'FAILURE.txt').write_text(traceback.format_exc());status();dump(ROOT/'FAILURE_COST.json',{'counts':dict(counts),'total_units':sum(counts.values()),'GPU_seconds':0 if GPU_START is None else time.monotonic()-GPU_START});(ROOT/'EXIT_CODE').write_text('1\\n');raise
    finally:done.set();ledger.close()
if __name__=='__main__':main()
'''
    start=text.index('        schedule=[]\n');end=text.index("        phase='building';step=0;GPU_START=time.monotonic();status()")
    for seed in [21701,21702]:
        seeded=text.replace('21601',str(seed));tail=suffix.replace('SEED_VALUE',str(seed))
        # Preparation and training use the same augmentation function. Both preparations precede any fit.
        prep=seeded[:end]+"        done.set();dump(ROOT/'PREPARATION_COMPLETE.json',{'schedule_SHA':sha(ROOT/'TARGET_SOURCE_SCHEDULE.json'),'units':0,'GPU_seconds':0});(ROOT/'EXIT_CODE').write_text('0\\n');return\n"+tail[tail.index('    except BaseException:'):]
        prep=prep.replace("'REPLACE_ROOT'",f"'preparation_seed{seed}'")
        if seed==21701:
            marker='        schedule=[]\n'
            replay="""        old_schedule=js(PARENT/'PHASE216_TARGET_SOURCE_SCHEDULE.json')
        for i,row in enumerate(old_schedule):
            _,proof=augment(row['index'],row['data_seed'])
            assert proof['image_SHA']==row['image_SHA'] and proof['mask_SHA']==row['mask_SHA'] and proof['draws']==row['draws']
            if i%500==0:step=i;phase='verify216_inputs';status()
        dump(PARENT/'PHASE216_INPUT_REPLAY_ACCEPTED.json',{'old_target_inputs_checked':4000,'new_neural_calls':0})
        phase='prepare217_sequence'
"""
            prep=prep.replace(marker,replay+marker)
        load="""        schedule=js(ROOT/'TARGET_SOURCE_SCHEDULE.json')
        locked=js(ROOT/'TRAINING_CONFIG_LOCK.json');assert sha(ROOT/'TARGET_SOURCE_SCHEDULE.json')==locked['schedule_SHA']
        original_source_sequence=js(PARENT/'SOURCE_SEQUENCE_LOCK.json')
        assert [{'source_sample_id':r['source_sample_id'],'source_png_name':r['source_png_name']} for r in schedule]==original_source_sequence
        assert len(schedule)==4000
"""
        train=(seeded[:start]+load+seeded[end:]+tail).replace("'REPLACE_ROOT'",f"'seed{seed}'")
        for label,code in [('prepare',prep),('train',train)]:
            code=code.replace(r"write_text('0\\n')",r"write_text('0\n')").replace(r"write_text('1\\n')",r"write_text('1\n')")
            ast.parse(code);name=f'{label}_phase217_seed{seed}.py';(OUT/'code'/name).write_text(code,encoding='utf-8')
    proof={'phase216_manifest_SHA':sha(OLD/'DELIVERY_MANIFEST.json'),'verified_old_artifacts':len(manifest),'training_kernel_SHA':sha(original),'generated_code_SHA':{p.name:sha(p) for p in (OUT/'code').glob('*.py')},'expected_units':48893,'breakdown':{'four_target_fits':16000,'two_source_backbone_teacher_student_streams':24000,'two_gates':10,'nonzero_candidate_evaluations':8880,'common_Source_parity_checks':3},'Source256_reselected':False,'seeds':[21701,21702]}
    (OUT/'LOCAL_PREFLIGHT.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
