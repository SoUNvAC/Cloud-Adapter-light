"""Freeze metadata-only scene split; require independently audited provenance.

Input inventory rows: scene, new_split, usgs_shadows, image_path, label_path.
Lineage JSON: used_scenes, sealed_scenes, unused_verified_scenes,
             provenance_sources (nonempty paths to metadata-only audit documents).
No imagery, labels, sealed predictions or test metrics are opened by this tool.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def allowed(path):
    resolved = Path(path).resolve()
    roots = [Path(__file__).resolve().parents[2]]
    if roots[0] == Path('/home/scv/Cloud-Adapter-light'):
        roots.append(Path('/home/scv/shared'))
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f'Path outside authorized roots: {path}')
    return resolved


def digest(path):
    h = hashlib.sha256()
    with allowed(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(rows, lineage):
    used, sealed, verified = [set(lineage[key]) for key in
                              ('used_scenes', 'sealed_scenes', 'unused_verified_scenes')]
    if not used or not sealed or not lineage['provenance_sources']:
        raise ValueError('Missing historical/sealed/provenance inventory')
    if verified & (used | sealed):
        raise ValueError('Verified unused scenes overlap historical or sealed scenes')
    selected = []
    for row in rows:
        if row['new_split'] != 'target_train':
            continue
        # Fail closed; only canonical yes or explicit 1 are admissible.
        if row['usgs_shadows'].strip().lower() not in ('yes', '1'):
            continue
        if row['scene'] in verified:
            selected.append(row.copy())
    scenes = sorted({r['scene'] for r in selected},
                    key=lambda s: hashlib.sha256(('phase65:' + s).encode()).hexdigest())
    if len(scenes) < 16:
        raise ValueError(f'Need >=16 independently unused eligible scenes, got {len(scenes)}')
    confirmation = set(scenes[:8])
    for row in selected:
        row['phase65_split'] = 'confirmation' if row['scene'] in confirmation else 'development'
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--lineage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory, lineage_path, out = map(allowed, (args.inventory, args.lineage, args.output))
    lineage = json.loads(lineage_path.read_text(encoding='utf-8'))
    source_hashes = {str(allowed(p)): digest(p) for p in lineage['provenance_sources']}
    with inventory.open(newline='', encoding='utf-8-sig') as stream:
        selected = freeze(list(csv.DictReader(stream)), lineage)
    for row in selected:
        for field in ('image_path', 'label_path'):
            allowed(row[field])  # including symlink resolution; no pixel read
    payload = dict(status='split_frozen_not_training_ready', rows=selected,
                   inventory_sha256=digest(inventory), lineage_sha256=digest(lineage_path),
                   provenance_sha256=source_hashes,
                   preregistration_sha256=digest(Path(__file__).resolve().parents[1] /
                                                  'research_plans/PHASE65.md'))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
