"""Final local acceptance, honest resource ledger and SHA manifest."""
from pathlib import Path
import csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase213/phase213_20261021';SUMMARY=ROOT/'src/research_results/phase213_20261021'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    for p in [OUT/'ANALYSIS_ACCEPTANCE.json',OUT/'RECEIVE_ACCEPTANCE.json',OUT/'REFERENCE_CPU_ACCEPTANCE.json',OUT/'delivery/REMOTE_CPU_ACCEPTANCE.json']:
        assert read(p)['status']=='passed'
    original=OUT/'ALPHA_SELECTION_LOCK.json';assert sha(original)=='53dc42499c8c2f1af238804cf0e25b8328598f64146f81d2c0a536de544516c9'
    launch=read(OUT/'remote_receipt/LAUNCH.json');assert sha(ROOT/'src/tools/execute_phase213_locked.py')==launch['code_sha256']
    raw=read(OUT/'delivery/RUN_COMPLETE.json');assert raw['calls']==11420 and raw['fits']==0
    ref=read(OUT/'REFERENCE_CPU_ACCEPTANCE.json');assert ref['new_model_calls']==0
    diagnosis=read(OUT/'STATE_DICT_DIAGNOSIS.json');assert diagnosis['mismatches']==[] and diagnosis['forward_calls']==0
    env=read(OUT/'FINAL_COST_IDENTITY.json');assert env['minimum_free_bytes']>=4*1024**3
    failure4=read(OUT/'remote_receipt/PRIOR_FAILURES.json');assert all(x['calls']==0 for x in failure4)
    # run03 timing is preserved in its stderr evidence and the separately retrieved diagnosis.
    failed_gate_seconds=4.897403552022297;construction_probe_seconds=diagnosis['elapsed_seconds']
    local_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    summary_bytes=sum(p.stat().st_size for p in SUMMARY.rglob('*') if p.is_file())
    remote=env['total_remote_snapshot_bytes']
    # Reserve an explicit 2MiB margin for final tiny receipts/manifests added after snapshots.
    upper=local_bytes+summary_bytes+remote+2*1024**2;assert upper<=4*1024**3
    cost={'new_neural_calls_total':11420,'source_recheck':535,'fixed_target_calls':10885,'all_failed_attempts_neural_calls':0,'extra_initialization_probe_neural_calls':0,'new_fits':0,'new_optimizer_updates':0,'new_image_or_weight_download_bytes':0,'successful_run_wall_seconds':raw['elapsed_seconds'],'failed_run03_gate_wall_seconds':failed_gate_seconds,'state_diagnosis_build_wall_seconds':construction_probe_seconds,'sum_recorded_model_stage_wall_seconds':raw['elapsed_seconds']+failed_gate_seconds+construction_probe_seconds,'wall_vs_GPU':'conservative wall-stage measurements, not pure GPU kernel timings','first_import_failure_wall_seconds':None,'first_import_failure_GPU_peak_bytes':None,'unobserved_failure_resource_fields':'unknown; no forward/model construction before missing-lock exit; do not substitute zero peak','successful_GPU_allocated_peak_bytes':raw['GPU_peak_allocated_bytes'],'successful_process_RSS_peak_bytes':raw['RSS_peak_bytes'],'physical_GPU_total_peak_bytes':None,'physical_GPU_total_peak_reason':'only PyTorch allocation peak was sampled, not total process/context or device-wide peak','CPU_reference_audit_seconds':ref['elapsed_seconds'],'remote_snapshot_bytes':remote,'local_snapshot_bytes':local_bytes,'summary_snapshot_bytes':summary_bytes,'combined_upper_including_finalization_margin_bytes':upper,'persistent_cap_bytes':4*1024**3,'host_minimum_final_free_bytes':env['minimum_free_bytes'],'host_sampling':'initial and final only; continuous minimum unknown','call_cap':12000,'GPU_stage_wall_cap_seconds':10800,'all_artifacts_small_text_no_new_weights':True}
    save(OUT/'COST_RECEIPT.json',cost)
    save(OUT/'EXECUTION_RECOVERY_LOCK.json',dict(launch,original_selection_lock_sha256=sha(original),successful_run='/home/scv/shared/phase213_20261021_run04',all_prior_attempts_preserved=True,original_EXECUTION_LOCK_describes_failed_first_attempt=True))
    doc=OUT/'RESULT.md';extra=f"\n## 3．最终交付验收\n\nCRC及逐文件SHA全部通过；独立标签LUT和NumPy floor-nearest格网核验覆盖535源图、2167 L8 test图、10 SPARCS test图，ignore均0，参考分母与每张混淆行和完全一致。所有10000次抽样均经另一种逐类IoU计算核对，并以排序插值独立复核区间分位数。\n\n计入已记录故障/诊断构建的阶段墙钟合计{cost['sum_recorded_model_stage_wall_seconds']:.3f}秒；首次仅导入模块的故障时间和峰值未采样，如实记unknown。GPU分配峰值不是设备/上下文总峰值。持久文件跨本地、远端、包及摘要的保守合计上界{upper/1024**2:.2f}MiB（含2MiB最终化余量），远低于4GiB；宿主固定物理卷的最小起止采样余量约{env['minimum_free_bytes']/1024**3:.2f}GiB，未连续采样。详情见COST_RECEIPT.json、FINAL_COST_IDENTITY.json。\n\n原始锁和失败目录未改写，未自动push、清理、同步work_dirs或启动下一phase。L8参考完整性和历史同支持test缺项保持公开。\n"
    assert '## 3．最终交付验收' not in doc.read_text(encoding='utf-8');doc.write_text(doc.read_text(encoding='utf-8')+extra,encoding='utf-8')
    for n in ['RESULT.md','COST_RECEIPT.json','REFERENCE_CPU_ACCEPTANCE.json','TEST_LABEL_METADATA_AUDIT.json','EXECUTION_RECOVERY_LOCK.json']:
        shutil.copyfile(OUT/n,SUMMARY/n)
    (OUT/'HANDOFF.md').write_text('# Phase213 — COMPLETE, 2026-10-23\n\nFixed run04 EXIT0,11420 calls,zero fits; Source535 exact recheck,all five alphas on L8 2167 inputs/16 scenes and SPARCS10 inputs/10 scenes completed.\n\nAll CPU arithmetic, SHA/CRC, label/ignore/reference denominator, grouped bootstrap and budget-selection acceptance passed. Source selection original SHA remains53dc42499c8c2f1af238804cf0e25b8328598f64146f81d2c0a536de544516c9.\n\nDelivery:RESULT.md,data and conclusions separate,all candidates/three budgets,all groups/confusions/error flows,CIs,cost/unknowns/failures/metadata limitations. Conclusion:SPARCS supported,L8 uncertain; ten L8 Shadows=no scenes limit reference reliability. Do not restart,change alpha/filter test,train,push,clean,or start another phase.\n',encoding='utf-8')
    codefiles=list((ROOT/'src/tools').glob('*phase213*.py'))+[ROOT/'src/tools/analyze_phase212_retrained.py',ROOT/'src/research_plans/PHASE213_SOURCE_BUDGET_ALPHA_INDEPENDENT_TEST_20261021.md']
    save(OUT/'CODE_HASHES.json',{'local_code_and_protocol':{str(p.relative_to(ROOT)):sha(p) for p in sorted(codefiles)},'actual_remote_runtime_helpers':env['remote_code_identity'],'actual_execution_code_sha256':launch['code_sha256']})
    shutil.copyfile(OUT/'CODE_HASHES.json',SUMMARY/'CODE_HASHES.json')
    manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json']}
    save(OUT/'DELIVERY_MANIFEST.json',manifest)
    save(OUT/'DELIVERY_COMPLETE.json',{'status':'complete','phase':'213','neural_calls':11420,'fits':0,'manifest_sha256':sha(OUT/'DELIVERY_MANIFEST.json'),'artifacts':len(manifest),'selected_status':'只在一个target-test成立','supported_target':'SPARCS','L8_status':'gain_CI_crosses_zero_and_annotation_completeness_limitation','no_next_phase_started':True})
    shutil.copyfile(OUT/'DELIVERY_COMPLETE.json',SUMMARY/'DELIVERY_COMPLETE.json')
    # Verify the final receipt references, not merely file existence.
    assert all((OUT/n).stat().st_size==r['bytes'] and sha(OUT/n)==r['sha256'] for n,r in manifest.items())
    print(json.dumps(read(OUT/'DELIVERY_COMPLETE.json'),ensure_ascii=False))
if __name__=='__main__':main()
