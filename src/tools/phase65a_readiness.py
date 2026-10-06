"""Audit existing L8 metadata against actual Phase64 training/validation lineage.

This inventories scene IDs from the split manifest, without opening test pixels.
It cannot certify new unseen data; existing Phase64 target_train was used to train.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from phase65a_freeze import allowed, digest


def read_csv(path):
    with allowed(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--shadow-metadata', type=Path, required=True)
    p.add_argument('--exploration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = read_csv(a.manifest)
    status = {r['scene']: r['usgs_shadows'] for r in read_csv(a.shadow_metadata)}
    exploration = {r['scene'] for r in read_csv(a.exploration)}
    splits = defaultdict(set)
    patch_counts = defaultdict(int)
    for row in rows:
        if status.get(row['scene']) != 'yes':
            continue
        splits[row['new_split']].add(row['scene'])
        patch_counts[row['new_split']] += 1
    if not exploration or not exploration <= splits['target_val']:
        raise ValueError('Historical exploratory scene lineage mismatch')
    # Phase64 target_train was already fitted. Treat all such scenes as used;
    # never claim fresh confirmation merely by repartitioning them.
    used = splits['target_train'] | splits['target_val'] | exploration
    sealed = splits['target_test']
    candidates = splits['target_train'] - used - sealed
    report = dict(status='blocked_no_independent_confirmation' if len(candidates) < 16
                  else 'requires_unused_provenance_verification',
                  shadow_filter='yes_only', shadow_valid_scene_counts={k: len(v) for k, v in splits.items()},
                  shadow_valid_patch_counts=dict(patch_counts),
                  historical_exploration_scenes=sorted(exploration),
                  previously_trained_scenes=sorted(splits['target_train']),
                  sealed_scene_count=len(sealed), eligible_independent_scene_count=len(candidates),
                  training_started=False, test_pixels_read=False,
                  interpretation='New unexposed scenes and audited usage history required; do not repartition old training as unseen confirmation.',
                  input_sha256={str(allowed(path)): digest(path) for path in
                                (a.manifest, a.shadow_metadata, a.exploration)})
    out = allowed(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
