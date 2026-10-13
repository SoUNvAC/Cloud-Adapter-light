"""Public aggregate/lock/SHA acceptance only; never opens local controlled labels."""
import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'src/research_results/phase103_20261013'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report=read(BASE/'support_report.json');lock=read(BASE/'prelabel_lock.json')
    assert sha(BASE/'reference_mapping.json')==lock['mapping_sha256']
    assert sha(BASE/'prelabel_lock.json')==report['prelabel_lock_sha256']
    product=rows(BASE/'per_product_support.csv');group=rows(BASE/'per_group_support.csv')
    assert len(product)==16 and len(group)==10
    assert {r['product_id'] for r in product}=={m['product_id'] for m in lock['members']}
    assert {r['group_id'] for r in group}=={m['group_id'] for m in lock['members']}
    for k in ['raw_points','deduplicated_positions','valid_positive','valid_surface_negative','valid_cloud_negative','valid_negative_total','ignored_positions','position_pending','20m_cells','20m_positive_negative_conflict_cells']:
        assert sum(int(r[k]) for r in product)==sum(int(r[k]) for r in group)==report['total_support'][k]
    for g in group:
        subset=[r for r in product if r['group_id']==g['group_id']]
        for k in ['raw_points','valid_positive','valid_negative_total','ignored_positions']:
            assert sum(int(r[k]) for r in subset)==int(g[k])
    assert report['AP_computable_groups']==sum(int(g['valid_positive'])>0 and int(g['valid_negative_total'])>0 for g in group)==8
    assert not report['AP_or_recall_or_FPR_computed'] and report['new_training_fits_predictions_imagery_downloads']==0
    receipt=read(BASE/'CPU_INDEPENDENT_VERIFIED.json');assert receipt['status']=='passed' and receipt['independent_truth_table_rows']==9500
    manifest=BASE/'DELIVERY_SHA256.json';count=None
    if manifest.exists():
        content=read(manifest)
        for name,h in content.items():assert sha(BASE/name)==h,name
        count=len(content)
    print(json.dumps({'status':'passed_aggregates_and_hashes_only','products':16,'groups':10,'sample_AP_supported_groups':8,
        'raw_and_derived_point_classes_opened':False,'model_metrics_computed':False,'payload_files':count,
        'manifest_sha256':sha(manifest) if manifest.exists() else None}))

if __name__=='__main__':main()
