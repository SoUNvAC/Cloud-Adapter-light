"""Package only sealed point scores and small CPU receipts from shared, no pixels."""
import json
import argparse
import os
from pathlib import Path
import zipfile
from phase118_common import ROOT, sha, load, write
from install_phase111_runtime import host_free


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--recover',action='store_true');args=parser.parse_args()
    if args.recover:ROOT=ROOT/'recovery_001'
    assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    pid=int((ROOT/'RUN_PID').read_text())
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        pass
    else:
        stat=Path(f'/proc/{pid}/stat').read_text()
        assert stat.split(') ')[1].split()[0]=='Z','worker still alive'
    state=load(ROOT/'RUN_STATUS.json')
    assert state['status']=='all_scores_sealed_before_local_reference_join'
    assert state['all_calls']<=(25000 if state.get('recovery') else 20000)
    assert state['elapsed_seconds']<=(18000 if state.get('recovery') else 14400)
    assert state['rss_peak_bytes']<=24*1024**3 and state['gpu_reserved_peak_bytes']<=20*1024**3
    assert state['repeat_calls']==8 and max(state['repeat_max_abs'])<=1e-5
    receipts=load(ROOT/'SCENE_RECEIPTS.json');seal=load(ROOT/'SCORES_SEALED.json')
    assert len(receipts)==32
    paths=[];point_counts={'C20':0,'H10':0};checks=[]
    for key,h in seal['score_sha256'].items():
        p=ROOT/'scores'/key/'UNLABELLED_POINT_SCORES.json';assert sha(p)==h
        mode=key.split('/')[0];point_counts[mode]+=len(load(p)['points'])
        c=p.with_name('CPU_POINT_RECONSTRUCTION.json');v=load(c)
        assert v['max_confidence_error']<=1e-5 and v['ensemble_max_error']<=1e-5 and v['coverage_and_point_gradient_exact']
        paths.extend([p,c]);checks.append({'setting_product':key,'max_confidence_error':v['max_confidence_error'],
                                         'ensemble_max_error':v['ensemble_max_error'],'coverage_exact':True})
    assert point_counts=={'C20':9500,'H10':9500}
    free=host_free();size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert free>=4000000000 and size<=4000000000
    write(ROOT/'CPU_ALL_SCENES_ACCEPTANCE.json',{'status':'passed','scenes':checks,'no_CPU_model_calls':True})
    write(ROOT/'COST_SNAPSHOT.json',{'host_free_after_bytes':free,'new_intermediate_bytes_before_package':size,
         'main_call_total':state['all_calls'],'main_run_wall_seconds':state['elapsed_seconds'],
         'single_GPU_wall_time_is_conservative_upper_bound_not_kernel_timer':True,
         'reserved_recovery_budget_used':state.get('recovery',False),'previous_attempt_calls':state.get('previous_attempt_calls',0),
         'no_download_fit_training':True})
    with zipfile.ZipFile(ROOT/'point_scores_private.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT/'scores').as_posix())
    small=[p for p in ROOT.iterdir() if p.suffix=='.json']
    write(ROOT/'SMALL_DELIVERY_SHA256.json',{p.name:sha(p) for p in small})
    with zipfile.ZipFile(ROOT/'small_delivery.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in small+[ROOT/'SMALL_DELIVERY_SHA256.json']:z.write(p,p.name)
    write(ROOT/'DELIVERY_RECEIPT.json',{'private_scores_archive_sha256':sha(ROOT/'point_scores_private.zip'),
         'private_scores_archive_bytes':(ROOT/'point_scores_private.zip').stat().st_size,
         'small_archive_sha256':sha(ROOT/'small_delivery.zip'),'small_archive_bytes':(ROOT/'small_delivery.zip').stat().st_size,
         'labels_accessed_or_transferred':False,'dense_array_files_transferred':False,'work_dirs_transferred':False})
    print(json.dumps(load(ROOT/'DELIVERY_RECEIPT.json')))


if __name__=='__main__':main()
