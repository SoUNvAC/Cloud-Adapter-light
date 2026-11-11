"""Two Source-only numerical parity probes; no fitting or candidate evaluation."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path('/home/scv/shared/phase216_20261023');SRC=Path('/home/scv/Cloud-Adapter-light/src');BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02')
def main():
    assert (ROOT/'EXIT_CODE').read_text().strip()=='1'
    assert not Path('/proc/'+(ROOT/'PID').read_text().strip()).exists()
    assert not (ROOT/'EVAL_NUMERICAL_DIAGNOSIS.json').exists()
    os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
    import numpy as np,torch
    from export_phase12_onnx import build_wrapper
    from validate_phase12_onnxruntime import load_rgb_image
    from evaluate_phase64_source_only_parent import load_parent_target,SOURCE_TO_PARENT
    old=json.loads((BASE/'interpolation/Source__source.json').read_text());fresh=json.loads((ROOT/'evaluation/C_a0__source.json').read_text());lookup={r['name']:r for r in old['per_input']}
    row=next(r for r in fresh['per_input'] if r['confusion']!=lookup[r['name']]['confusion']);name=row['name']
    wrapper,_=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'),precision='fp32',active_block_indices=None))
    x=torch.from_numpy(load_rgb_image(SRC/'data/cloudsen12_high_l1c/img_dir/val'/name,512)).cuda();y=load_parent_target(SRC/'data/cloudsen12_high_l1c/ann_dir/val'/name,SOURCE_TO_PARENT,(512,512));valid=y!=255
    prior=[json.loads(s) for s in (ROOT/'CALL_LEDGER.jsonl').read_text().splitlines()];n=len(prior);assert n==20540
    ledger=(ROOT/'CALL_LEDGER.jsonl').open('a',buffering=1);start=time.monotonic();out=[]
    for deterministic in [True,False]:
        torch.use_deterministic_algorithms(deterministic,warn_only=True);torch.backends.cudnn.deterministic=deterministic;torch.backends.cudnn.benchmark=False
        n+=1;assert n<=30000;ledger.write(json.dumps({'ordinal':n,'kind':'diagnostic_Source_full_segmentor','stage':'numerical_parity_diagnosis','deterministic':deterministic,'step':0})+'\n')
        with torch.inference_mode():
            s=wrapper(x).float().clamp_min(0);a=s.argmax(1);b=(s/s.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)
            results={}
            for key,p in [('raw',a),('legacy_normalized',b)]:
                p=p[0].cpu().numpy();c=np.bincount(3*y[valid]+p[valid],minlength=9).reshape(3,3).tolist();results[key]={'confusion':c,'matches_historical':c==lookup[name]['confusion'],'matches_failed_fresh':c==row['confusion']}
            changed=a!=b;v=s.topk(2,dim=1).values;margin=v[:,0]-v[:,1]
            out.append({'deterministic':deterministic,'results':results,'argmax_changes_due_to_FP32_normalization':int(changed.sum()),'max_raw_top2_margin_on_changed_pixels':float(margin[changed].max()) if changed.any() else None})
    ledger.close()
    result={'selected_input':name,'selection':'first numerical-confusion mismatch, engineering parity only','probes':out,'additional_calls':2,'cumulative_units':n,'seconds':time.monotonic()-start,'model_refits':0,'canonical_evaluation':'original helper normalization, torch default deterministic=False/cudnn.deterministic=False','cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    (ROOT/'EVAL_NUMERICAL_DIAGNOSIS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
