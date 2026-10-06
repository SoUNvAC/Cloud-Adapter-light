"""Freeze group-disjoint cohorts and public H1 support before any new prediction.

Metadata lock is NOT training authorization. Source file SHA must be verified on
the training machine; pipeline/model/statistical implementation gates remain.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from phase65a_freeze import allowed, digest

SPLIT_SIZES = [('65a_confirmation', 64), ('65b_confirmation', 64),
               ('65c_final', 64), ('development_val', 32)]


class SupportGateError(ValueError):
    def __init__(self, rows, counts):
        self.rows, self.counts = rows, counts
        super().__init__(f'Insufficient preregistered H1 support: {counts}; do not resample')


def make_split(groups, tags):
    tagged = {r['scene']: r for r in tags}
    eligible = [g for g in groups if g['eligible']]
    if len(eligible) < 240:
        raise ValueError('Need 224 reserved/validation groups plus >=16 fit groups')
    eligible.sort(key=lambda g: hashlib.sha256(('phase65:split:' + g['group_id']).encode()).hexdigest())
    assigned, used_products, used_members = [], set(), set()
    boundaries = []
    cursor = 0
    for name, size in SPLIT_SIZES:
        boundaries.append((name, cursor, cursor + size))
        cursor += size
    boundaries.append(('fit', cursor, len(eligible)))
    for split, begin, end in boundaries:
        for group in eligible[begin:end]:
            if group['exclusion_reasons'] or used_members & set(group['members']):
                raise ValueError('Excluded or overlapping group in proposed split')
            used_members.update(group['members'])
            products = group['shadow_valid_products']
            expected = {p for p in group['members'] if tagged[p]['shadows_marked'] == '1'}
            if not products or set(products) != expected or used_products & set(products):
                raise ValueError('Invalid/duplicate shadow-valid samples')
            used_products.update(products)
            representative = min(products, key=lambda p: hashlib.sha256(
                ('phase65:representative:' + p).encode()).hexdigest())
            percent = float(tagged[representative]['shadow_percent'])
            if not math.isfinite(percent) or not 0 <= percent <= 100:
                raise ValueError('Unknown public H1 support')
            assigned.append(dict(group_id=group['group_id'], split=split,
                                 products=sorted(products), representative=representative,
                                 h1_supported=percent > 0))
    counts = {split: sum(row['h1_supported'] for row in assigned if row['split'] == split)
              for split in ('fit', '65a_confirmation')}
    if any(count < 16 for count in counts.values()):
        raise SupportGateError(assigned, counts)
    return assigned


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--tags', type=Path, required=True)
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    groups = json.loads(allowed(a.groups).read_text())
    if groups['status'] != 'groups_prepared_not_split_locked':
        raise ValueError('Group provenance audit incomplete')
    if digest(a.tags) not in groups['input_sha256'].values():
        raise ValueError('Catalogue metadata SHA mismatch')
    actual = digest(a.source_checkpoint)
    if actual != a.source_sha256:
        raise ValueError('Source checkpoint SHA mismatch')
    with allowed(a.tags).open(newline='', encoding='utf-8') as stream:
        tags = list(csv.DictReader(stream))
    try:
        rows = make_split(groups['groups'], tags)
    except SupportGateError as error:
        # Preserve the precise failed proposal, without training authorization or
        # a replacement random draw. It must never be mistaken for a passed lock.
        output = allowed(a.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(status='stopped_insufficient_h1_support', training_authorized=False,
                      h1_support_counts=error.counts, required_each=16,
                      proposed_rows=error.rows, source_checkpoint_sha256=actual,
                      group_sha256=digest(a.groups), target_tags_sha256=digest(a.tags),
                      preregistration_sha256=digest(Path(__file__).resolve().parents[1] /
                                                   'research_plans/PHASE65.md'),
                      new_model_predictions_read=False, target_pixels_read=False)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != 'proposed_rows'}))
        raise
    summary = {split: dict(groups=sum(r['split'] == split for r in rows),
                           products=sum(len(r['products']) for r in rows if r['split'] == split),
                           h1_support=sum(r['h1_supported'] for r in rows if r['split'] == split))
               for split in [name for name, _ in SPLIT_SIZES] + ['fit']}
    prereg = Path(__file__).resolve().parents[1] / 'research_plans/PHASE65.md'
    protocol = dict(rows=rows, source_checkpoint_sha256=actual, target_tags_sha256=digest(a.tags),
                    group_sha256=digest(a.groups), preregistration_sha256=digest(prereg),
                    target_seed=65, target_steps=4000, target_lr=1e-4,
                    classes=['surface_visible', 'cloud', 'shadow'],
                    source_checkpoint=str(allowed(a.source_checkpoint)))
    canonical = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    report = dict(status='metadata_split_frozen_not_training_ready', protocol=protocol,
                  protocol_sha256=canonical, summary=summary, training_authorized=False,
                  target_pixels_read=False, new_model_predictions_read=False,
                  pending=['complete_two_machine_acquisition', 'array_and_loader_audit',
                           'training_config_and_statistical_code_verification'])
    output = allowed(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'protocol'}))


if __name__ == '__main__':
    main()
