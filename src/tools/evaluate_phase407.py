"""Inference-only residual scaling; Phase400 artifacts remain read-only."""
import os, sys, json, time, hashlib, csv, traceback, shutil, subprocess,threading,resource
from pathlib import Path
OLD = Path('/home/scv/shared/phase400_20261031')
OUT = Path(os.environ['PHASE407_EVAL_ROOT'])
STATE={'calls':0,'updates':0,'current':None};START=time.monotonic()
sys.path.insert(0, str(OLD/'code'))
from phase400_common import setup, build, config, SOURCE, sha, js, state_sha, hostfree

def save(name, value):
    p = OUT/name
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(p)

def table(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def main():
    start=time.monotonic(); calls=0
    assert not (OUT/'RUN_INFO.json').exists()
    hostfree(); assert shutil.disk_usage(OUT).free>10*1024**3
    smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    gpu=subprocess.check_output([smi,'--query-compute-apps=pid','--format=csv,noheader'], text=True)
    assert not gpu.strip(), ('GPU already occupied',gpu)
    lock=js(OUT/'INPUT_EXPECTED.json')
    assert sha(OUT/'REFERENCE_GROUPS.csv')==lock['groups_SHA']
    assert sha(Path(__file__))==lock['worker_SHA']
    for relative, expected in lock['old_files'].items(): assert sha(OLD/relative)==expected,relative
    code_lock=js(OLD/'CODE_LOCK.json')
    for relative, expected in code_lock['existing_dependencies_LF'].items():
        raw=(Path('/home/scv/Cloud-Adapter-light')/relative).read_bytes()
        assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==expected,relative
    ck=Path(os.environ['PHASE407_CHECKPOINT'])
    assert sha(ck)==lock['H_SHA'] and sha(SOURCE)==lock['Source_SHA']
    torch=setup()
    import numpy as np
    from mmseg.registry import DATASETS
    from mmengine.dataset import pseudo_collate
    from report_phase400 import describe, ap_exact
    os.environ['PHASE65_PREPARED_MANIFEST']=str(OLD/'PREPARED_MANIFEST.json')
    os.environ['PHASE65_SOURCE_CHECKPOINT']=str(SOURCE)
    model,cfg,_=build('H',None)
    model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
    model.eval(); assert not model.backbone.target_msre.enabled
    identity=state_sha(model.state_dict())
    def count(m,args,kwargs):
        nonlocal calls
        calls+=1;STATE.update(calls=calls,seconds=time.monotonic()-start)
        with (OUT/'CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps(STATE)+'\n')
        assert calls<=855,'normal planned full-input calls exceeded; no automatic recovery'
        assert time.monotonic()-start<3600,'1h total runtime budget'
    call_hook=model.register_forward_pre_hook(count,with_kwargs=True)
    stop=threading.Event()
    def guard():
        while not stop.wait(15):
            try:
                assert time.monotonic()-start<3600
                assert shutil.disk_usage(OUT).free>10*1024**3
                assert torch.cuda.max_memory_reserved()<.9*torch.cuda.get_device_properties(0).total_memory
            except BaseException:
                save('FAILURE.json',{'traceback':traceback.format_exc(),'state':STATE,'seconds':time.monotonic()-start});(OUT/'EXIT_CODE').write_text('1');os._exit(71)
    threading.Thread(target=guard,daemon=True).start()
    alpha=[0.0]*4
    hooks=[h.register_forward_hook(lambda m,a,out,slot=slot:out*alpha[slot]) for slot,h in enumerate(model.backbone.target_head_delta)]
    inp=js(OLD/'INPUT_LOCK.json')
    target=DATASETS.build(cfg.val_dataloader.dataset)
    source=DATASETS.build(dict(type='Phase64CloudSENParentDataset',data_root=inp['source_val_root'],data_prefix=dict(img_path='img_dir/val',seg_map_path='ann_dir/val'),pipeline=[dict(type='LoadImageFromFile'),dict(type='Resize',scale=(512,512)),dict(type='LoadAnnotations'),dict(type='PackSegInputs')]))
    assert len(target)==32 and len(source)==535
    source_map={r['name']:r for r in inp['source_val']}
    target_map={r['product']:r for r in inp['target_rows'] if r['split']=='development_val'}
    for record in target_map.values():
        for field,suffix in [('rgb_path','_rgb.npy'),('mask_path','_mask.npy'),('image_valid_path','_image_valid.npy')]:
            assert sha(record[field])==record['prepared_sha256'][suffix]
    with (OUT/'REFERENCE_GROUPS.csv').open() as f:
        reference={r['name']:r for r in csv.DictReader(f) if r['case']=='0.5'}
    assert len(reference)==32
    summaries=[]; groups=[]; images=[]; endpoint_checks=[]; cases={}
    def schedule():
        cases['reference_0.5']=[.5]*4
        yield 'reference_0.5'
        lo,hi=lock['bracket']
        target_value=lock['target_reference']
        for i in range(8):
            name=f'probe_{i:02d}';mid=(lo+hi)/2;cases[name]=[mid]*4
            yield name
            value=summaries[-1]['target_mIoU']
            if value<target_value:lo=mid
            else:hi=mid
        selected=min((r for r in summaries if r['case'].startswith('probe_')),key=lambda r:(abs(r['target_mIoU']-target_value),cases[r['case']][0]))
        cases['selected']=cases[selected['case']][:]
        save('SELECTION.json',{'case':selected['case'],'alpha':cases['selected'][0],'target_mIoU':selected['target_mIoU'],'target_reference':target_value,'absolute_error':abs(selected['target_mIoU']-target_value),'source_calls_before_selection':0,'calls_before_selection':calls,'rule':'minimum absolute target error, tie smaller alpha; exactly8 probes'})
        yield 'selected'
    for av in schedule():
        alpha[:]=cases[av]; totals={}; aps=[]; all_scores=[]; all_labels=[]
        for domain,dataset in ([('target',target),('source',source)] if av=='selected' else [('target',target)]):
            aggregate=np.zeros((3,3),np.int64)
            old_report=None
            for i in range(len(dataset)):
                assert time.monotonic()-start<3600
                sample=dataset[i]; name=Path(sample['data_samples'].img_path).name
                if domain=='source':
                    record=source_map[name]
                    assert sha(record['image'])==record['image_SHA'] and sha(record['mask'])==record['mask_SHA']
                else:
                    name=name.removesuffix('_rgb.npy');record=target_map[name]
                STATE['current']={'case':str(av),'domain':domain,'index':i,'name':name}
                with torch.inference_mode(): pred=model.test_step(pseudo_collate([sample]))[0]
                truth=sample['data_samples'].gt_sem_seg.data[0].numpy(); valid=truth!=255
                prediction=pred.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
                assert np.bincount(truth[valid],minlength=3).tolist()==record['label_counts' if domain=='target' else 'parent_counts']
                cm=np.bincount(3*truth[valid].astype(np.int64)+prediction[valid],minlength=9).reshape(3,3)
                aggregate+=cm; digest=hashlib.sha256(prediction.tobytes()).hexdigest()
                row={'case':str(av),'name':name,**describe(cm),'prediction_SHA':digest}
                if av=='reference_0.5':
                    assert json.loads(reference[name]['confusion'])==cm.tolist() and reference[name]['prediction_SHA']==digest
                if av=='selected' and domain=='target':
                    selected_case=js(OUT/'SELECTION.json')['case']
                    prior=next(x for x in groups if x['case']==selected_case and x['name']==name)
                    assert prior['confusion']==cm.tolist() and prior['prediction_SHA']==digest
                if domain=='target':
                    logits=pred.seg_logits.data.float().clamp_min(0); probs=logits/logits.sum(0,keepdim=True).clamp_min(1e-12)
                    assert torch.isfinite(probs).all()
                    scores=probs[2].cpu().numpy()[valid].astype(np.float64); labels=truth[valid]==2
                    ap=ap_exact(scores,labels);aps.append(ap);all_scores.append(scores);all_labels.append(labels)
                    row.update(group_id=record['group_id'],AP=None if ap is None else ap*100)
                    groups.append(row)
                else: images.append(row)
                if i%50==0:
                    assert torch.cuda.max_memory_reserved()<.9*torch.cuda.get_device_properties(0).total_memory
                    save('STATUS.json',{'case':str(av),'domain':domain,'index':i,'calls':calls,'seconds':time.monotonic()-start})
                    assert shutil.disk_usage(OUT).free>10*1024**3
                    if time.monotonic()-STATE.get('last_host_check',0)>180:hostfree();STATE['last_host_check']=time.monotonic()
            totals[domain]=describe(aggregate)

        assert state_sha(model.state_dict())==identity
        summaries.append({'case':str(av),'alpha':cases[av][0],**{f'{d}_{k}':v for d in totals for k,v in totals[d].items()},'target_macro_AP':float(np.mean([a for a in aps if a is not None])*100),'target_pooled_AP':ap_exact(np.concatenate(all_scores),np.concatenate(all_labels))*100,'AP_groups':sum(a is not None for a in aps)})
        assert summaries[-1]['AP_groups']==7
        table('summary.csv',summaries)
    save('ENDPOINT_GATE.json',{'passed':True,'reference_target_predictions':32,'selected_target_predictions_reproduced':32})
    for h in hooks:h.remove()
    call_hook.remove();stop.set()
    assert calls==855
    table('per_group.csv',groups);table('source_per_image.csv',images)
    assert sum(1 for _ in (OUT/'CALL_LEDGER.jsonl').open())==calls
    save('RUN_INFO.json',{'passed':True,'calls':calls,'updates':0,'seconds':time.monotonic()-start,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'CPU_main_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'checkpoint_SHA':sha(ck),'Source_SHA':sha(SOURCE),'state_SHA_unchanged':identity,'endpoint_checks':endpoint_checks,'cases':cases,'script_SHA':sha(Path(__file__)),'source_validation_selection':False,'ledger_SHA':sha(OUT/'CALL_LEDGER.jsonl'),'input_expected_SHA':sha(OUT/'INPUT_EXPECTED.json'),'labels_definition':['surface','cloud','shadow'],'status':'complete'})

if __name__=='__main__':
    try:
        main();(OUT/'EXIT_CODE').write_text('0')
    except BaseException:
        save('FAILURE.json',{'traceback':traceback.format_exc(),'state':STATE,'seconds':time.monotonic()-START});(OUT/'EXIT_CODE').write_text('1');raise
