"""Read JSON metadata only; never open target arrays, archives or predictions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/home/scv/Cloud-Adapter-light')
DATA = Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    split_path = REPO/'src/work_dirs/phase65a/split_lock.json'
    split = json.loads(split_path.read_text())
    assert sha(split_path) == 'a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b'
    rows = split['proposed_rows']
    selected = [r for r in rows if r['split'] == '65c_final']
    products = {p for r in selected for p in r['products']}
    others = {p for r in rows if r['split'] != '65c_final' for p in r['products']}
    assert len(selected) == 64 and not products & others
    scanned, hits, errors, provenance = {}, [], [], []
    # Limit reads to recognized access metadata, never metric reports or score arrays.
    folders = list(DATA.glob('phase*')) + [REPO/'src/work_dirs']
    for folder in sorted(folders):
        if not folder.is_dir():
            continue
        paths = set(folder.rglob('manifest.json')) | set(folder.rglob('*index*.json')) | set(folder.rglob('unseal_receipt.json'))
        for p in sorted(paths):
            try:
                raw = p.read_bytes()
                text = raw.decode('utf-8')
                json.loads(text)
                scanned[str(p)] = hashlib.sha256(raw).hexdigest()
                overlap = sorted(product for product in products if product in text)
                if overlap:
                    hits.append(dict(path=str(p), products=overlap))
            except Exception as e:
                errors.append(dict(path=str(p), error=str(e)))
    for p in (REPO/'src/work_dirs/phase65a/source_provenance').glob('*.json'):
        if sha(p) != split['group_sha256']:
            continue
        gd = json.loads(p.read_text())
        groups = {g['group_id']:g for g in gd['groups']}
        for r in selected:
            g = groups[r['group_id']]
            assert g['eligible'] and set(r['products']) == set(g['shadow_valid_products'])
        dependencies = []
        for name, expected in gd['input_sha256'].items():
            dep = Path(name)
            assert any(dep.resolve().is_relative_to(root) for root in (REPO, Path('/home/scv/shared')))
            dependencies.append(dict(path=name, expected_sha256=expected, actual_sha256=sha(dep)))
        provenance.append(dict(path=str(p), sha256=sha(p), selected_groups_eligible=True,
                               dependencies=dependencies,
                               selected_groups=[groups[r['group_id']] for r in selected]))
    dependency_ok = bool(provenance) and all(d['expected_sha256']==d['actual_sha256'] for x in provenance for d in x['dependencies'])
    print(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='metadata_checked_not_authorization_to_unseal', split_sha256=sha(split_path),
        selected_rows=selected, group_count=len(selected), product_count=len(products),
        public_h1_support=sum(r['h1_supported'] for r in selected),
        product_disjoint_from_other_splits=True, scanned_metadata=scanned,
        prior_access_hits=hits, scan_errors=errors, provenance=provenance,
        provenance_dependencies_match=dependency_ok, pixels_read=False,
        limitation='Retrospective manifests cannot exclude undocumented/manual reads; independence is conditional on recorded lineage and conservative footprint audit.',
        ready_for_unsealing=False), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
