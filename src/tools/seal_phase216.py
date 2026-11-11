"""Final seal only after numerical and human/AI visual QA; no remote actions."""
from pathlib import Path
import hashlib,json,time,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase216/phase216_20261023';SUMMARY=ROOT/'src/research_results/phase216_20261023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert js(OUT/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed'
    assert js(OUT/'FIGURE_VISUAL_QA.json')['status']=='passed'
    assert not (OUT/'DELIVERY_COMPLETE.json').exists()
    receipt=js(OUT/'REMOTE_DELIVERY_RECEIPT.json');assert 'remote_scope_snapshot_bytes' in receipt
    for n in ['RESULT.md','ALL_CANDIDATES.csv','JOINT_RULE.csv','PAIRED_CI.csv','FINITE_SET_DOMINANCE.csv','INDEPENDENT_CPU_ACCEPTANCE.json','LOSS_SUMMARY.json','LOSS_CURVES.png','LOSS_CURVES.pdf','LOSS_CURVES.svg','FIGURE_VISUAL_QA.json','SOURCE_SPATIAL_METADATA_AUDIT.json']:
        shutil.copy2(OUT/n,SUMMARY/n)
    local=size(OUT);summary=size(SUMMARY);code={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/tools').glob('*phase216*.py')};codebytes=sum((ROOT/n).stat().st_size for n in code)
    remote=receipt['remote_scope_snapshot_bytes']+receipt['remote_archive_bytes'];total=remote+local+summary+codebytes;assert total<4_000_000_000 and shutil.disk_usage(ROOT).free>4*1024**3
    save(OUT/'FINAL_COST_CODE_RECEIPT.json',{'epoch':time.time(),'remote_scope_and_archive_bytes':remote,'local_scope_bytes_before_final_seal':local,'summary_bytes_before_final_seal':summary,'new_code_bytes':codebytes,'combined_snapshot_bytes':total,'byte_cap':4_000_000_000,'manifest_and_final_receipt_pending_small_bytes':True,'disk_free_local_bytes':shutil.disk_usage(ROOT).free,'code_SHA':code,'GPU_cost':js(OUT/'delivery/COMPLETE.json'),'CPU_analysis':js(OUT/'ANALYSIS_RECEIPT.json'),'CPU_report':js(OUT/'LOCAL_REPORT_COST.json'),'remote_CPU_acceptance_peak_bytes':receipt['CPU_acceptance_peak_bytes'],'no_source_weights_or_large_arrays_transferred':True,'no_automatic_push_or_timer_or_next_phase':True,'historical_code_hash_scope':'Historical four helpers checked, current code snapshot retained; no claim unrecorded historical code hashes are known'})
    shutil.copy2(OUT/'FINAL_COST_CODE_RECEIPT.json',SUMMARY/'FINAL_COST_CODE_RECEIPT.json')
    handoff=OUT/'HANDOFF.md';handoff.write_text('# Phase216 complete\n\nBoth4000-step arms, all five fixed alphas/common Source, source-only selection lock before target metrics, full tables/bootstrap and independent acceptance complete. See RESULT.md. No next phase, no automation, no push. Original artifacts and failed engineering attempts preserved.\n',encoding='utf-8')
    save(OUT/'LOCAL_STATUS.json',{'stage':'complete','numeric_acceptance':'passed','visual_QA':'passed','new_model_units':25537,'no_running_GPU_or_receiver':True})
    exclude={'DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json'}
    manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and not (p.parent==OUT and p.name in exclude)}
    save(OUT/'DELIVERY_MANIFEST.json',manifest)
    assert all((OUT/n).stat().st_size==r['bytes'] and sha(OUT/n)==r['sha256'] for n,r in manifest.items())
    finaltotal=remote+size(OUT)+size(SUMMARY)+codebytes;assert finaltotal<4_000_000_000
    save(OUT/'DELIVERY_COMPLETE.json',{'status':'complete','epoch':time.time(),'artifacts':len(manifest),'manifest_SHA':sha(OUT/'DELIVERY_MANIFEST.json'),'all_files_reread_verified':True,'combined_snapshot_bytes_before_this_receipt':finaltotal,'new_model_calls':js(OUT/'delivery/COMPLETE.json')['total_units']})
    print(json.dumps({'complete':True,'artifacts':len(manifest),'new_bytes':finaltotal}))
if __name__=='__main__':main()
