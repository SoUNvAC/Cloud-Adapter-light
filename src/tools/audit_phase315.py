"""Independent CPU ledger marginal audit, without refitting, forward or new pool access."""
from collections import defaultdict
from datetime import datetime,timezone
import resource
import time
from phase315_common import *
def main():
    started=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));assert load(ROOT/'CPU_ACCEPTANCE.json')['status']=='passed';assert not (ROOT/'INDEPENDENT_ACCEPTANCE.json').exists()
    metrics={(r['seed'],r['product']):r for r in read(ROOT/'PER_GROUP_METRICS.csv')};tensors=defaultdict(list)
    for r in read(ROOT/'TRANSITIONS27.csv'):tensors[r['seed'],r['product']].append(r)
    events={(r['seed'],r['product']):r for r in read(ROOT/'D_B_PER_GROUP.csv')};flows={(r['seed'],r['product'],r['error']):r for r in read(ROOT/'ERROR_FLOW.csv')};checks=0
    C=['Surface','Cloud','Shadow']
    for (seed,product),rows in tensors.items():
        assert len(rows)==27;old=[[0]*3 for _ in range(3)];new=[[0]*3 for _ in range(3)];t={}
        for r in rows:
            a,b,c=C.index(r['truth']),C.index(r['Source_pred']),C.index(r['student_pred']);v=int(r['count']);old[a][b]+=v;new[a][c]+=v;t[a,b,c]=v
        if product!='POOLED32':
            assert old==json.loads(metrics['Source',product]['confusion']) and new==json.loads(metrics[seed,product]['confusion']);e=events[seed,product];assert int(e['D'])==t[0,0,1] and int(e['B'])==t[1,2,1]
        for a in range(3):
            for b in range(3):
                if a==b:continue
                row=flows[seed,product,C[a]+'->'+C[b]];oldn=old[a][b];newn=new[a][b];same=t[a,b,b]
                assert int(row['Source_errors'])==oldn and int(row['student_errors'])==newn
                assert int(row['removed'])==oldn-same and int(row['added'])==newn-same and int(row['net_reduction'])==oldn-newn
                assert int(row['corrected'])==t[a,b,a] and int(row['changed_to_other_error'])==oldn-same-t[a,b,a];checks+=1
    assert sum(int(r['D']) for r in events.values() if r['seed']=='30201')==506798 and sum(int(r['B']) for r in events.values() if r['seed']=='30201')==784478
    for r in read(ROOT/'EVENT_OVERLAP.csv'):
        old,new,intersection,union=map(lambda k:int(r[k]),['old','new','intersection','union']);assert old+new-intersection==union and int(r['old_only'])==old-intersection and int(r['new_only'])==new-intersection
        assert sum(int(r[k]) for k in ['old_to_Surface','old_to_Cloud','old_to_Shadow'])==old
        assert (r['Jaccard']=='NA') if not union else abs(float(r['Jaccard'])-intersection/union)<1e-12;checks+=1
    identities=read(ROOT/'SEED_IDENTITY.csv');assert len(identities)==3 and len({r['initial_A_sha256'] for r in identities})==3 and all(r['B_zero']=='True' for r in identities)
    for seed in ['31501','31502']:
        ep=load(ROOT/seed/'ENDPOINT_LOCK.json');gate=load(ROOT/seed/'MODEL_ACCEPTANCE.json');assert ep['updates']==955 and ep['skipped']==45 and ep['all_base_tensors_unchanged']
        assert sha(ROOT/seed/'initial_lora.pth')==ep['initial_sha256'] and sha(ROOT/seed/'final_lora.pth')==ep['final_sha256'] and sha(ROOT/seed/'TRAIN_CURVE.jsonl')==ep['curve_sha256']
        for name,h in ep['probes'].items():assert sha(ROOT/seed/'probes'/name)==h,name
        assert max(gate['same_seed_original304_AST_loss_error'],gate['same_seed_gradient_max_error'],gate['same_seed_update_max_error'],*gate['zero_lora_errors'].values())<=1e-6
    for path,h in load(ROOT/'CPU_INPUT_SHA256.json').items():assert sha(path)==h,path
    for path,h in load(ROOT/'EXECUTION_LOCK.json')['inputs'].items():assert sha(path)==h,path
    calls=[json.loads(line) for line in (ROOT/'CALLS.jsonl').read_text().splitlines()];assert len(calls)<=2800
    write(ROOT/'INDEPENDENT_ACCEPTANCE.json',dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),checks=checks,all96_group_tensors_and3pooled_exact=True,Source_new_confusions_and6flows_reproduced=True,historical_D0=506798,historical_B0=784478,all_overlap_margins_exact=True,three_distinct_initial_A=True,B_initial_zero=True,original_step_gates_passed=True,all_original_input_cache_SHA_reverified=True,actual_forwards=len(calls),raw_truth_or_model_arrays_reread=False,seconds=time.monotonic()-started,peak_cpu_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print('Phase315 independent margins passed;',len(calls),'forwards')
if __name__=='__main__':main()
