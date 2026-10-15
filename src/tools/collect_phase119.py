"""Collect aggregate results and private PixBox scores from shared only."""
from pathlib import Path
import zipfile
from phase119_common import ROOT,load,write,sha
from install_phase111_runtime import host_free


def main():
    assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    state=load(ROOT/'CONTROLLER_STATUS.json');assert state['calls_completed']<=4000 and state['elapsed_seconds']<=7200
    assert load(ROOT/'Catalogue_CPU_ACCEPTANCE.json')['status']=='passed'
    # Producer must have exited before final immutable packaging.
    pid=(ROOT/'PID').read_text().strip();p=Path('/proc')/pid/'stat'
    assert not p.exists() or p.read_text().split(') ')[1].split()[0]=='Z'
    costs=[];models=[];actual=0
    private=[];small=[ROOT/'INPUT_PATCH_PLAN.json',ROOT/'ALL_SCORES_SEALED.json',ROOT/'CONTROLLER_STATUS.json',
                      ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json',ROOT/'Catalogue_CPU_ACCEPTANCE.json']
    small+=list((ROOT/'aggregate').iterdir())
    for policy in ('E','R','N'):
        d=ROOT/policy;s=load(d/'RUN_STATUS.json');assert s['status']=='completed_scores_sealed'
        assert s['rss_peak_bytes']<=16*1024**3 and s['gpu_reserved_peak_bytes']<=20*1024**3
        actual+=s['all_calls'];models.append(load(d/'MODEL_INFO.json'))
        for q in load(d/'SCENE_RECEIPTS.json'):
            assert q['max_confidence_error']<=1e-5 and q['max_raw_mean_error']<=1e-5 and q['coverage_gradient_exact']
            if 'old_E_max_error' in q:assert q['old_E_max_error']<=1e-5
            if 'repeat_full_array_max_error' in q:assert q['repeat_full_array_max_error']<=1e-5
        small += [d/n for n in ('RUN_STATUS.json','MODEL_INFO.json','SCENE_RECEIPTS.json','SCORES_SEALED.json')]
        private+=list((d/'scores/PixBox').glob('*/UNLABELLED_POINT_SCORES.json'))
        for q in (d/'scores/PixBox').glob('*/UNLABELLED_POINT_SCORES.json'):
            assert sha(q)==load(d/'SCORES_SEALED.json')['scores_sha256']['PixBox/'+q.parent.name]
    assert len(private)==35 and actual==state['calls_completed']
    from phase119_common import LOCK
    bench=load(LOCK)['benchmark_products']
    for policy in ('E','R','N'):
        records=load(ROOT/policy/'SCENE_RECEIPTS.json')
        for r in records:
            if r['pool']=='PixBox' and r['product_id'] in bench:
                costs.append({'policy':policy,'product':r['product_id'],'repetition':r['repetition'],
                              'GPU_model_seconds':r['model_GPU_seconds'],
                              'whole_scene_read_to_point_seconds':r['read_to_point_seconds_excluding_SHA_and_independent_validation'],
                              'model_calls':r['model_calls']})
    assert len(costs)==18
    write(ROOT/'BENCHMARK_COSTS.json',{'rows':costs,'model_info':models,'all18_results_reported':True,
          'three_separate_processes':True,'fresh_E_forward_not_cache_timing':True,
          'SHA_CPU_independent_validation_and_packaging_excluded':True,'OS_cache_order_effect_not_eliminated':True,
          'fixed_process_order':['E','R','N'],'fixed_scene_order':bench})
    write(ROOT/'COST_SUMMARY.json',{'actual_model_calls':actual,'inference_wall_seconds':state['elapsed_seconds'],
          'host_physical_free_bytes':host_free(),'new_intermediate_bytes_before_packaging':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
          'no_training_fitting_new_download_or_threshold_search':True,'no_work_dirs_sync':True,
          'workers':{p:load(ROOT/p/'RUN_STATUS.json') for p in ('E','R','N')}})
    assert host_free()>=4000000000
    small += [ROOT/'BENCHMARK_COSTS.json',ROOT/'COST_SUMMARY.json']
    with zipfile.ZipFile(ROOT/'point_scores_private.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in private:z.write(p,p.relative_to(ROOT).as_posix())
    write(ROOT/'SMALL_DELIVERY_SHA256.json',{p.relative_to(ROOT).as_posix():sha(p) for p in small})
    with zipfile.ZipFile(ROOT/'small_delivery.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in small+[ROOT/'SMALL_DELIVERY_SHA256.json']:z.write(p,p.relative_to(ROOT).as_posix())
    write(ROOT/'DELIVERY_RECEIPT.json',{'private_archive_bytes':(ROOT/'point_scores_private.zip').stat().st_size,
          'private_archive_sha256':sha(ROOT/'point_scores_private.zip'),'small_archive_bytes':(ROOT/'small_delivery.zip').stat().st_size,
          'small_archive_sha256':sha(ROOT/'small_delivery.zip'),'labels_transferred':False,'dense_arrays_transferred':False,
          'Catalogue_private_cache_transferred':False,'work_dirs_transferred':False})
    print(load(ROOT/'DELIVERY_RECEIPT.json'))


if __name__=='__main__':main()
