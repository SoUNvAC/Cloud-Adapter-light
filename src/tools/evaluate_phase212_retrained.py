"""Fixed Phase212 FP32 Source/head interpolation. No fitting or saved logits."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

REPO=Path('/home/scv/Cloud-Adapter-light');SRC=REPO/'src'
ROOT=Path('/home/scv/shared/phase212_retrain_20261020_run02')
OUT=ROOT/'interpolation'
SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
SOURCE_SHA='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=str));t.replace(p)

def main():
    os.chdir(SRC);sys.path.insert(0,str(SRC));sys.path.insert(0,str(SRC/'tools'));os.environ['XFORMERS_DISABLED']='1'
    import numpy as np
    import torch
    from export_phase12_onnx import build_wrapper
    from validate_phase12_onnxruntime import load_rgb_image
    from evaluate_phase64_source_only_parent import paired_directory_records,manifest_records,l8_shadow_yes_records,load_parent_target,SOURCE_TO_PARENT,L8_TO_PARENT,SPARCS_TO_PARENT
    assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    assert sha(SOURCE)==SOURCE_SHA
    OUT.mkdir(exist_ok=False);(OUT/'heads').mkdir()
    # Extract only scoped numerical references; do not copy/sync historical work_dirs reports.
    history={'MsRE':{},'LoRA':{},'source_files':{},'precision':'historical FP16, distinct from fixed FP32 new runs'}
    oldroot=SRC/'work_dirs/phase64_rgb_recheck'
    switch=oldroot/'source_switch.json'
    if switch.is_file():
        old=json.loads(switch.read_text());history['source_files'][str(switch)]=sha(switch)
        for target in ['l8','sparcs']:
            for label,key in [('MsRE','msre'),('LoRA','lora')]:
                history[label].setdefault(target,{})['source_metrics_exact']=old['conditions'][target][key]['adaptation_branch_enabled']['metrics']
    paired=oldroot/'paired_scene.json'
    if paired.is_file():
        old=json.loads(paired.read_text());history['source_files'][str(paired)]=sha(paired)
        for target in ['l8','sparcs']:
            history['MsRE'].setdefault(target,{})['target_metrics_exact']=old['domains'][target]['msre']['metrics']
    for target,value in [('l8',62.80),('sparcs',68.82)]:
        history['LoRA'].setdefault(target,{})['target_mIoU_rounded_only']=value
    dump(OUT/'HISTORICAL_REFERENCES.json',history)
    clock=time.monotonic();calls=0;ledger=(OUT/'CALL_LEDGER.jsonl').open('x',buffering=1)
    completed={};headproof={};receipts={t:json.loads((ROOT/t/'COMPLETED.json').read_text()) for t in ['l8','sparcs']}
    source_state=torch.load(SOURCE,map_location='cpu')['state_dict']
    for t in receipts:
        assert sha(receipts[t]['head_path'])==receipts[t]['head_sha256']
        audit=json.loads((ROOT/t/'FROZEN_FINAL_AUDIT.json').read_text());assert not audit['mismatches']
    source_root=SRC/'data/cloudsen12_high_l1c'
    if not source_root.is_dir():source_root=REPO/'data/cloudsen12_high_l1c'
    rows={'source':paired_directory_records(source_root,'val'),'l8':l8_shadow_yes_records(SRC/'work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv',SRC/'research_plans/protocol_data/l8_biome_usgs_shadow_status.csv','target_val'),'sparcs':manifest_records(SRC/'work_dirs/phase64_sparcs_protocol_verified_v2/scene_disjoint_manifest.csv','target_val')}
    assert [len(rows[n]) for n in ['source','l8','sparcs']]==[535,963,10]
    groups={x['png_name']:x for x in csv.DictReader((ROOT/'eval_code/SOURCE_ROI_BINDINGS.csv').open())}
    assert len(groups)==535 and len({x['roi_id'] for x in groups.values()})==107
    for row in rows['source']:
        assert sha(row['image_path'])==groups[row['name']]['image_sha256']
        row['scene']=groups[row['name']]['roi_id']
    maps={'source':SOURCE_TO_PARENT,'l8':L8_TO_PARENT,'sparcs':SPARCS_TO_PARENT}
    plan={pool:[dict(r,image_sha256=sha(r['image_path']),mask_sha256=sha(r['mask_path'])) for r in rs] for pool,rs in rows.items()}
    dump(OUT/'INPUT_SUPPORT_LOCK.json',{'pools':plan,'source_ROIs':107,'target_scenes':{'l8':8,'sparcs':10},'input_size':512,'source_groups':'official ROI, historical PNG identity mapping verified; not535 independent scenes','class_mapping':{p:maps[p].tolist() for p in maps},'ignore':255,'target_test_read':False,'source_test_read':False})
    def build(precision='fp32'):
        w,c=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SOURCE),precision=precision,active_block_indices=None))
        for p in w.parameters():p.requires_grad=False
        assert not w.training
        return w
    wrapper=build()
    param_names={n for n,_ in wrapper.segmentor.named_parameters() if n.startswith('decode_head.')}
    buffer_names={n for n,_ in wrapper.segmentor.named_buffers() if n.startswith('decode_head.')}
    full=dict(list(wrapper.segmentor.named_parameters())+list(wrapper.segmentor.named_buffers()))
    mismatches=[n for n,t in source_state.items() if n not in full or not torch.equal(t,full[n].detach().cpu())]
    assert not mismatches,mismatches[:10]
    frozen={n:t.detach().cpu().clone() for n,t in full.items() if not n.startswith('decode_head.')}
    source_head={n:t.detach().cpu().clone() for n,t in full.items() if n.startswith('decode_head.')}
    targets={}
    for target in receipts:
        ck=torch.load(receipts[target]['head_path'],map_location='cpu');head=ck['state_dict'];assert set(head)==set(source_head)
        assert ck['phase212_reconstruction']['Source_SHA']==SOURCE_SHA
        assert not [n for n in buffer_names if not torch.equal(head[n],source_head[n])]
        assert all(head[n].shape==source_head[n].shape and head[n].dtype==source_head[n].dtype for n in head)
        assert all(torch.is_floating_point(source_head[n]) for n in param_names)
        targets[target]=head
        headproof[target]={'head_sha256':receipts[target]['head_sha256'],'parameter_entries':len(param_names),'buffer_entries':len(buffer_names),'buffer_mismatches':[],'all_non_head_equal_to_Source_during_training':True,'full_Source_reconstruction_exact_tensors':len(source_state),'trainable_parameters':1316516,'new_retraining':True}
    dump(OUT/'PARAMETER_BUFFER_ACCEPTANCE.json',headproof)
    def loadhead(h):
        state=wrapper.segmentor.state_dict();state.update(h);wrapper.segmentor.load_state_dict(state,strict=True)
        now=dict(list(wrapper.segmentor.named_parameters())+list(wrapper.segmentor.named_buffers()))
        assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
        assert all(torch.equal(t,now[n].detach().cpu()) for n,t in h.items())
    def evaluate(name,pool):
        nonlocal calls
        total=np.zeros((3,3),np.int64);per_group=collections.defaultdict(lambda:np.zeros((3,3),np.int64));per_input=[];predhash=hashlib.sha256()
        with torch.inference_mode():
            for i,row in enumerate(rows[pool]):
                assert calls<16000 and time.monotonic()-clock<14400
                rgb=torch.from_numpy(load_rgb_image(Path(row['image_path']),512)).cuda()
                calls+=1;ledger.write(json.dumps({'call':calls,'endpoint_or_alpha':name,'pool':pool,'input':i,'elapsed_seconds':time.monotonic()-clock})+'\n')
                scores=wrapper(rgb).float().clamp_min(0)
                probs=scores/scores.sum(1,keepdim=True).clamp_min(1e-12)
                assert torch.isfinite(probs).all()
                pred=probs.argmax(1)[0].cpu().numpy().astype(np.uint8);predhash.update(pred.tobytes())
                y=load_parent_target(Path(row['mask_path']),maps[pool],pred.shape);valid=y!=255
                cm=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3)
                total+=cm;per_group[row['scene']]+=cm
                per_input.append({'name':row['name'],'group':row['scene'],'confusion':cm.tolist()})
                if i%100==0 or i+1==len(rows[pool]):
                    dump(OUT/'RUN_STATUS.json',{'stage':'evaluating','model':name,'pool':pool,'completed_inputs':i+1,'pool_inputs':len(rows[pool]),'calls':calls,'elapsed_seconds':time.monotonic()-clock})
        result={'model':name,'pool':pool,'confusion':total.tolist(),'per_group':{k:v.tolist() for k,v in sorted(per_group.items())},'per_input':per_input,'prediction_sha256':predhash.hexdigest(),'inputs':len(rows[pool]),'groups':len(per_group)}
        dump(OUT/(name+'__'+pool+'.json'),result);completed[(name,pool)]=result
        return result
    def miou(cm):
        x=np.asarray(cm,dtype=np.float64);tp=np.diag(x);u=x.sum(0)+x.sum(1)-tp
        return 100*float(np.nanmean(np.divide(tp,u,out=np.full(3,np.nan),where=u>0)))
    try:
        for pool in ['source','l8','sparcs']:evaluate('Source',pool)
        for target in ['l8','sparcs']:
            loadhead(targets[target])
            for pool in ['source',target]:evaluate(target+'_a1',pool)
        expected={'source':77.8179,'l8':58.0530,'sparcs':55.5775}
        actual={p:miou(completed[('Source',p)]['confusion']) for p in expected}
        match={p:round(actual[p],4)==round(v,4) for p,v in expected.items()}
        endpoint={'new_targets_are_retrained_not_old_checkpoint':True,'Source_FP32':actual,'historical_Source_FP16':expected,'Source_round4_matches':match,'new_alpha1_target':{t:miou(completed[(t+'_a1',t)]['confusion']) for t in ['l8','sparcs']},'historical_precision_difference_diagnosis':None}
        if not all(match.values()):
            # Diagnose the documented historical FP16 wrapper. Keep FP32 main decision fixed.
            wrapper.cpu();del wrapper;torch.cuda.empty_cache();wrapper=build('fp16')
            diag={}
            for pool in ['source','l8','sparcs']:diag[pool]=miou(evaluate('Source_historical_fp16_diagnostic',pool)['confusion'])
            dm={p:round(diag[p],4)==round(expected[p],4) for p in expected}
            endpoint['historical_precision_difference_diagnosis']={'FP16':diag,'round4_match':dm,'additional_calls':1508,'FP32_main_not_changed':True}
            dump(OUT/'ENDPOINT_REPRODUCTION.json',endpoint)
            if not all(dm.values()):raise RuntimeError('Historical Source reproduction unresolved even with original FP16; stop before internal alphas')
            wrapper.cpu();del wrapper;torch.cuda.empty_cache();wrapper=build('fp32')
        dump(OUT/'ENDPOINT_REPRODUCTION.json',endpoint)
        for target in ['l8','sparcs']:
            for alpha in [0.25,0.5,0.75]:
                h={}
                for n,s in source_head.items():
                    t=targets[target][n]
                    h[n]=(s.to(torch.float64)+alpha*(t.to(torch.float64)-s.to(torch.float64))).to(s.dtype) if n in param_names else s.clone()
                    assert torch.isfinite(h[n]).all()
                    if n in param_names:assert torch.equal(h[n],(s.double()+alpha*(t.double()-s.double())).to(s.dtype))
                name=target+'_a'+str(alpha).replace('.','p')
                p=OUT/'heads'/(name+'.pth');torch.save({'state_dict':h,'Source_SHA':SOURCE_SHA,'target_SHA':receipts[target]['head_sha256'],'alpha':alpha,'formula_dtype':'float64 CPU then original dtype'},p)
                dump(OUT/'HEAD_HASHES.json',{p.name:sha(p) for p in (OUT/'heads').glob('*.pth')})
                loadhead(h)
                for pool in ['source',target]:evaluate(name,pool)
        allocated=torch.cuda.max_memory_allocated();assert allocated<=20*1024**3
        b=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert b<=500_000_000
        dump(OUT/'COMPLETED.json',{'status':'all_fixed_points_evaluated','calls':calls,'main_expected_calls':9680,'diagnostic_extra_calls':calls-9680,'elapsed_seconds':time.monotonic()-clock,'GPU_peak_allocated_bytes':allocated,'new_combined_persistent_bytes':b,'models_trained_here':0,'all_comparisons_held_fixed':True})
        (OUT/'EXIT_CODE').write_text('0\n')
    except BaseException:
        (OUT/'FAILURE.txt').write_text(traceback.format_exc());dump(OUT/'FAILURE_COST.json',{'calls':calls,'elapsed_seconds':time.monotonic()-clock,'completed_model_pools':[list(x) for x in completed]});(OUT/'EXIT_CODE').write_text('1\n');raise
    finally:ledger.close()

if __name__=='__main__':main()
