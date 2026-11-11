"""Independent CPU-only checkpoint arithmetic and call accounting; no inference."""
from pathlib import Path
import hashlib,json,collections
ROOT=Path('/home/scv/shared/phase213_20261021_run04');OUT=ROOT/'evaluation';BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    import torch
    torch.set_num_threads(2)
    assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    src=Path('/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')
    s=torch.load(src,map_location='cpu')['state_dict'];s={n:v for n,v in s.items() if n.startswith('decode_head.')}
    proofs=read(OUT/'PARAMETER_ACCEPTANCE.json');assert proofs['all_parameters_frozen'] and proofs['buffers_equal']
    checks=0
    for t in ['l8','sparcs']:
        th=proofs['heads'][t]['1'];assert sha(th['path'])==th['sha256'];target=torch.load(th['path'],map_location='cpu')['state_dict'];assert set(s)==set(target)
        for k in [1,2,3]:
            a=k/4;head=proofs['heads'][t][str(a)];assert sha(head['path'])==head['sha256'];h=torch.load(head['path'],map_location='cpu')['state_dict'];assert set(h)==set(s)
            for n,x in s.items():
                expected=((4-k)*x.double()+k*target[n].double()).div(4).to(x.dtype) if x.is_floating_point() else x
                assert h[n].shape==x.shape and h[n].dtype==x.dtype and torch.equal(h[n],expected),(t,a,n)
                checks+=1
    lines=[json.loads(x) for x in (OUT/'CALL_LEDGER.jsonl').read_text().splitlines()]
    assert [x['call'] for x in lines]==list(range(1,11421))
    counts=collections.Counter((r['pool'],r['target'],r['alpha']) for r in lines)
    assert counts['source','l8',0]==535
    for t,n in [('l8',2167),('sparcs',10)]:
        for a in [0,.25,.5,.75,1]:assert counts[t,t,a]==n
    admission=read(OUT/'FINAL_ADMISSION_LOCK.json');assert admission['calls_before_target']==535 and not admission['test_opened_before_seal']
    source=read(OUT/'source_a0.json');old=read(BASE/'interpolation/Source__source.json');assert source['confusion']==old['confusion'] and source['prediction_sha256']==old['prediction_sha256']
    complete=read(OUT/'RUN_COMPLETE.json');assert complete['calls']==len(lines)<=12000 and complete['elapsed_seconds']<=10800
    proof={'status':'passed','independent_weighted_sum_formula_tensor_checks':checks,'full_head_shapes_and_dtype_exact':True,'checkpoint_SHA_exact':True,'source_recheck_predictions_and_confusions_exact':True,'call_sequence_1_to_11420_exact':True,'grouped_call_counts_exact':True,'no_neural_calls_in_validator':True,'nonhead_freeze_evidence':'main evaluator strict tensor comparison before and after all model/pool evaluations; runtime proof, not extra forward','manifest_and_seal_prior_to_target':True,'cost':complete}
    (OUT/'REMOTE_CPU_ACCEPTANCE.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
if __name__=='__main__':main()
