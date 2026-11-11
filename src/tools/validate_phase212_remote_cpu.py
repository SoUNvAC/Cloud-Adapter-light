"""Independent CPU-only endpoint SHA/confusion and interpolation formula checks."""
from pathlib import Path
import hashlib
import json

ROOT=Path('/home/scv/shared/phase212_retrain_20261020_run02');OUT=ROOT/'interpolation'
SOURCE=Path('/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (OUT/'EXIT_CODE').read_text().strip()=='0'
    import torch
    source=torch.load(SOURCE,map_location='cpu')['state_dict'];sh={n:t for n,t in source.items() if n.startswith('decode_head.')}
    receipt=json.loads((ROOT/'RETRAINING_COMPLETE.json').read_text());formulas=[]
    locked=json.loads((OUT/'HEAD_HASHES.json').read_text())
    for target in ['l8','sparcs']:
        item=receipt['heads'][target];p=Path(item['head_path']);assert sha(p)==item['head_sha256']
        th=torch.load(p,map_location='cpu')['state_dict'];assert set(th)==set(sh)
        initial=json.loads((ROOT/target/'INITIAL_IDENTITY.json').read_text());final=json.loads((ROOT/target/'FROZEN_FINAL_AUDIT.json').read_text())
        assert initial['all_non_head_digest']==final['digest'] and not initial['source_mismatch'] and not final['mismatches']
        assert initial['source_checkpoint_tensors_exact']==381 and initial['trainable_head_parameters']==1316516
        for alpha in [.25,.5,.75]:
            path=OUT/'heads'/(target+'_a'+str(alpha).replace('.','p')+'.pth');assert sha(path)==locked[path.name]
            ck=torch.load(path,map_location='cpu');assert ck['Source_SHA']==item['Source_SHA'] and ck['target_SHA']==item['head_sha256']
            h=ck['state_dict'];assert set(h)==set(sh)
            for n in h:
                expected=torch.lerp(sh[n].double(),th[n].double(),alpha).to(sh[n].dtype) if torch.is_floating_point(sh[n]) else sh[n]
                assert torch.equal(h[n],expected),('independent_lerp',target,alpha,n)
                assert torch.isfinite(h[n]).all()
            formulas.append({'target':target,'alpha':alpha,'state_entries':len(h),'independent_CPU_lerp_exact':True,'sha256':sha(path)})
    baseline={}
    oldroot=Path('/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_rgb_recheck')
    switch=oldroot/'source_switch.json';targetbase=oldroot/'three_parent_target_baseline.json'
    if switch.is_file():baseline['source']=json.loads(switch.read_text())['source_parent_checkpoint']['evaluation']
    if targetbase.is_file():baseline.update(json.loads(targetbase.read_text())['domains'])
    exact=[]
    for pool,old in baseline.items():
        p=OUT/('Source_historical_fp16_diagnostic__'+pool+'.json')
        if not p.is_file():p=OUT/('Source__'+pool+'.json')
        new=json.loads(p.read_text());assert new['prediction_sha256']==old['prediction_sha256'],('historical_prediction_SHA',pool)
        assert new['confusion']==old['metrics']['confusion'],('historical_exact_confusion',pool)
        exact.append({'pool':pool,'prediction_SHA_exact':True,'confusion_exact':True,'matching_precision':'historical FP16 diagnostic' if 'diagnostic' in p.name else 'main FP32'})
    assert set(x['pool'] for x in exact)=={'source','l8','sparcs'},'all historical exact references required'
    result={'status':'passed_CPU_only','model_forward_calls':0,'training':0,'independent_head_formulas':formulas,'historical_Source_exact_reproduction':exact,'source_and_two_target_initial_final_non_head_equal':True,'source_buffer_and_param_identity':'381 checkpoint entries exact initialization;281 non-head named tensors frozen','evaluation_CPU_peak_bytes':'not continuously sampled; unknown','old_FP16_vs_new_FP32_difference':'explicit diagnostic; main FP32 endpoints retained'}
    (OUT/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
