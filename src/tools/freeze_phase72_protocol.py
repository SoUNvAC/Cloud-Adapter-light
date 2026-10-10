"""Freeze existing models and metadata only. This script cannot run evaluation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/phase72/phase72_20261010'
BUNDLE = ROOT/'src/research_plans/phase72_frozen_20261010'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    prior = ROOT/'outputs/phase71/phase71_20261010'
    policy_path = prior/'delivery/policy_lock.json'
    assert sha(policy_path) == 'ec409be4f85085c3cbdffdb0ab97ef0ec09b7c60d34351b397136dc7e19d834c'
    policy = read(policy_path)
    audit_path = OUT/'metadata_audit_expanded.json'
    audit = read(audit_path)
    assert not audit['prior_access_hits'] and not audit['scan_errors']
    assert audit['provenance_dependencies_match'] and not audit['pixels_read']
    split_path = ROOT/'outputs/phase65/work_dirs/phase65a/split_lock.json'
    assert sha(split_path) == audit['split_sha256']
    rows = read(split_path)['proposed_rows']
    selected = [r for r in rows if r['split']=='65c_final']
    assert selected == audit['selected_rows']
    spatial_path = ROOT/'outputs/phase65/work_dirs/phase65a/source_provenance/spatial_overlap.json'
    deps = {d['path']:d['actual_sha256'] for p in audit['provenance'] for d in p['dependencies']}
    assert sha(spatial_path) == deps['/home/scv/Cloud-Adapter-light/src/work_dirs/phase65a/source_provenance/spatial_overlap.json']
    boxes = {r['product']:r['buffered_wgs84_bounds'] for r in read(spatial_path)['findings']}
    candidates = [(p,r['group_id']) for r in selected for p in r['products']]
    others = [(p,r['split']) for r in rows if r['split']!='65c_final' for p in r['products']]
    def intersects(a,b):
        return a[0]<=b[2] and b[0]<=a[2] and a[1]<=b[3] and b[1]<=a[3]
    across = [(p,q,s) for p,g in candidates for q,s in others if intersects(boxes[p],boxes[q])]
    within = [(p,q) for i,(p,g) in enumerate(candidates) for q,h in candidates[i+1:]
              if g!=h and intersects(boxes[p],boxes[q])]
    assert not across and not within
    BUNDLE.mkdir(exist_ok=False)
    policy_sources = [prior/'delivery'/n for n in ['policy_lock.json','input_lock.json','feature_index.json','fit_input_audit.json','baseline_reproduction.json']]
    models = {k:policy['models'][k] for k in ['B','L3','RGB_local']}
    base = ['MsRE_logit','Source_logit','B08_TOA','B11_TOA','B12_TOA']
    features = {'L3':base,'B':base+['B08_local','B11_local','B12_local'],
                'RGB_local':base+['B04_local','B03_local','B02_local']}
    freeze = dict(status='models_and_protocol_frozen_evaluation_not_started',
        evaluation_authorized=False, evaluation_pool='original_65c_final_64_fixed_representatives',
        original_split_sha256=sha(split_path), rows=selected, models=models, features=features,
        model_canonical_sha256={k:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest() for k,v in models.items()},
        existing_input_locks={str(p.relative_to(ROOT)):sha(p) for p in policy_sources},
        original_input_lock=read(prior/'delivery/input_lock.json'),
        official_files=read(ROOT/'data/sentinel2_cloud_mask_catalogue_4172871/download_status.json')['verified_files'],
        fit_sampling='153 groups equal weight; max8192 pixels/group without replacement; seed65081+first8hex SHA256(product); original1253376 samples unchanged',
        fit_input_sha256=dict(source_index='51044a1ecc6d8a8c6e58802551ce78ad0ce205f2344235154b629fe2c1c4b534',
            msre_index='1b9fe63bc90383f92b62a09ee04f9f2164464b0fe606accf05eb7a5b0f3bf8e0',
            feature_index='0db1b08b1cd44be259de5970cfa9656c2134bd7e3058ab5864e4c0249dd19b34',
            sample_original_five='b5ceece265c7b46a8a2a768951aba57aaca7c7ec053b69c0359b6fda66c1e2cf',
            sample_labels='e659d5dc7a48c4ecd48bf5adad6cdb969b87800751744bb3beb3e00ff38a8a27'),
        source_checkpoint_sha256='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9',
        msre_checkpoint_sha256='82f88d235948ef0a9202cebaa026e7f9fb1e1ad5c0aa361d9045c3e6991ee16d',
        local_window=61,formula='(center-mean)/(abs(mean)+1e-6)',center_excluded=True,image_valid_neighbors_only=True,
        bootstrap=dict(seed=72010,resamples=10000,method='paired equal related-group percentile; two-sided AP 2.5/97.5; one-sided safety 5/95'),
        gates=dict(primary='B-L3 mean AP two-sided95 lower>0',secondary='B-RGB mean AP two-sided95 lower>0, tested only after primary and all safety gates pass',
                   B_mean_fpr_one_sided_upper95_max=.01, B_minus_L3_mean_recall_one_sided_lower95_min=-.01,
                   B_minus_L3_mean_miou_pp_one_sided_lower95_min=-1),
        user_confirmed_safety_margins=True, pixels_read=False,
        metadata_audit_sha256=sha(audit_path), metadata_scanned_files=len(audit['scanned_metadata']),
        public_shadow_support_groups=audit['public_h1_support'],
        spatial_audit=dict(input_sha256=sha(spatial_path),all_66_products_checked=True,
                           overlaps_other_fixed_splits=across,between_distinct_65c_groups=within,
                           limitation='Conservative buffered WGS84 bounds, conditional on original footprint metadata accuracy'),
        protocol_sha256=sha(ROOT/'src/research_plans/PHASE72_INDEPENDENT_CONFIRMATION_20261010.md'))
    write(BUNDLE/'confirmation_lock.json',freeze)
    table = list(csv.DictReader((prior/'all_group_changes.csv').open(encoding='utf-8')))
    row = next(r for r in table if r['tile']=='T32NKF')
    with (OUT/'T32NKF_noB12_delivery.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    write(OUT/'delivery_addendum.json',dict(source_table_sha256=sha(prior/'all_group_changes.csv'),
        source_policy_sha256=sha(policy_path),no_refit=True,row=row,
        frozen_protocol_lock_sha256=sha(BUNDLE/'confirmation_lock.json')))
    print(json.dumps(dict(models=list(models),groups=len(selected),public_shadow_groups=audit['public_h1_support'],
        metadata_files_scanned=len(audit['scanned_metadata']),spatial_overlaps=0,evaluation_started=False)))

if __name__=='__main__':
    main()
