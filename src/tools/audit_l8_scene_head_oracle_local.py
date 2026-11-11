"""Fixed two-head, whole-scene oracle audit using existing confusion only."""
from pathlib import Path
import csv
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/research_audits/l8_scene_head_oracle_20261028'
INPUT = ROOT / 'outputs/phase236/phase236_20261027/PER_SCENE.json'
MANIFEST = ROOT / 'src/research_plans/phase236_frozen_20261027/EVAL_INPUTS.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(cm):
    ious = []
    for k in range(3):
        union = sum(cm[k * 3:k * 3 + 3]) + sum(cm[j * 3 + k] for j in range(3)) - cm[4 * k]
        if union:
            ious.append(cm[4 * k] / union * 100)
    return sum(ious) / len(ious), len(ious)


def add_rows(rows):
    return [sum(row[k] for row in rows) for k in range(9)]


def flat(row):
    return [v for a in row['confusion'] for v in a]


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    identities = {str(p.relative_to(ROOT)): sha(p) for p in [INPUT, MANIFEST]}
    data = json.loads(INPUT.read_text(encoding='utf-8-sig'))
    rows = [x for x in data if x['domain'] == 'L8' and x['split'] == 'test']
    metadata = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))['L8']['test']
    strata = {x['scene']: x['usgs_shadows'] for x in metadata}
    source = {x['scene']: x for x in rows if x['strategy'] == 'Source' and x['source']}
    scenes = sorted(source)
    assert len(scenes) == 16 and set(scenes) == set(strata)
    n = len(scenes)
    summaries, per_scene, checks = [], [], []
    # Main pair fixed before outcomes: Source vs val-selected W (alpha=.75).
    # Secondary: Source vs actual trained endpoint C at the same fixed LR 1e-4.
    for head in ['W', 'C_endpoint']:
        for seed in [21701, 21702]:
            other = {x['scene']: x for x in rows if x['seed'] == seed and not x['source']
                     and x['strategy'] == ('W' if head == 'W' else 'endpoint')
                     and x['lr'] == 1e-4}
            assert len(other) == n and set(other) == set(scenes)
            assert all(other[s]['alpha'] == (.75 if head == 'W' else 1) for s in scenes)
            a = [flat(source[s]) for s in scenes]
            b = [flat(other[s]) for s in scenes]
            for i, scene in enumerate(scenes):
                assert [sum(a[i][k:k + 3]) for k in [0, 3, 6]] == [sum(b[i][k:k + 3]) for k in [0, 3, 6]]
                assert math.isclose(metrics(a[i])[0], source[scene]['mIoU'], abs_tol=1e-8)
                assert math.isclose(metrics(b[i])[0], other[scene]['mIoU'], abs_tol=1e-8)
            av, bv = [metrics(v)[0] for v in a], [metrics(v)[0] for v in b]
            scene_mask = sum((1 << i) for i in range(n) if bv[i] > av[i])
            def chosen(mask):
                return [b[i] if mask & (1 << i) else a[i] for i in range(n)]
            def score(mask):
                return metrics(add_rows(chosen(mask)))[0]
            # Enumerate all 2^16 whole-scene routes with Gray-code incremental sums.
            current = add_rows(a)
            deltas = [[b[i][j] - a[i][j] for j in range(9)] for i in range(n)]
            best_mask, best_value, previous = 0, metrics(current)[0], 0
            visited = 1
            for ordinal in range(1, 1 << n):
                mask = ordinal ^ (ordinal >> 1)
                changed = mask ^ previous
                index = changed.bit_length() - 1
                sign = 1 if mask & changed else -1
                current = [current[j] + sign * deltas[index][j] for j in range(9)]
                value = metrics(current)[0]
                if (value > best_value or (value == best_value and
                    (mask.bit_count(), mask) < (best_mask.bit_count(), best_mask))):
                    best_value, best_mask = value, mask
                previous = mask
                visited += 1
            assert visited == 65536
            assert math.isclose(score(best_mask), best_value, abs_tol=1e-10)
            assert best_value >= max(score(0), score((1 << n) - 1), score(scene_mask)) - 1e-10
            assert sum(max(x, y) for x, y in zip(av, bv)) / n >= max(sum(av) / n, sum(bv) / n) - 1e-10
            masks = [('all_Source', 0), ('all_adapted', (1 << n) - 1),
                     ('oracle_scene_mIoU', scene_mask), ('oracle_pooled_mIoU_exact', best_mask)]
            for strategy, mask in masks:
                cms = chosen(mask)
                summaries.append({'head': head, 'seed': seed, 'strategy': strategy,
                                  'pooled_mIoU': score(mask),
                                  'scene_mean_mIoU': sum(metrics(v)[0] for v in cms) / n,
                                  'adapted_scenes': mask.bit_count(), 'Source_scenes': n - mask.bit_count(),
                                  'pooled_gain_over_best_constant_pp': score(mask) - max(score(0), score((1 << n) - 1)),
                                  'scene_mean_gain_over_best_constant_pp': sum(metrics(v)[0] for v in cms) / n - max(sum(av) / n, sum(bv) / n),
                                  'reference_Shadow_pixels': sum(v[6] + v[7] + v[8] for v in cms),
                                  'route_bits_sorted_scenes': mask,
                                  'confusion_flat_row_major': json.dumps(add_rows(cms))})
            for i, scene in enumerate(scenes):
                per_scene.append({'head': head, 'seed': seed, 'scene': scene, 'shadows_metadata': strata[scene],
                                  'Source_mIoU': av[i], 'adapted_mIoU': bv[i],
                                  'adapted_minus_Source_pp': bv[i] - av[i],
                                  'Source_valid_classes': metrics(a[i])[1], 'adapted_valid_classes': metrics(b[i])[1],
                                  'oracle_scene_choice': 'adapted' if scene_mask & (1 << i) else 'Source',
                                  'oracle_pooled_choice': 'adapted' if best_mask & (1 << i) else 'Source',
                                  'reference_pixels': sum(a[i]), 'reference_Shadow_pixels': sum(a[i][6:9])})
            checks.append({'head': head, 'seed': seed, 'same_reference_support': True,
                           'scene_metrics_reproduced': True, 'exhaustive_routes': visited,
                           'pooled_optimum_fresh_sum_verified': True, 'dominates_all_constant_routes': True})
    # Average seed metrics only; never fuse predictions or routes between seeds.
    means = []
    for head in ['W', 'C_endpoint']:
        for strategy in ['all_Source', 'all_adapted', 'oracle_scene_mIoU', 'oracle_pooled_mIoU_exact']:
            rs = [x for x in summaries if x['head'] == head and x['strategy'] == strategy]
            assert len(rs) == 2
            row = {'head': head, 'strategy': strategy}
            row.update({k: sum(x[k] for x in rs) / 2 for k in [
                'pooled_mIoU', 'scene_mean_mIoU', 'adapted_scenes',
                'pooled_gain_over_best_constant_pp', 'scene_mean_gain_over_best_constant_pp']})
            means.append(row)
    write_csv('SUMMARY_PER_SEED.csv', summaries)
    write_csv('SUMMARY_SEED_METRIC_MEANS.csv', means)
    write_csv('PER_SCENE_CHOICES.csv', per_scene)
    write_json('CHECKS.json', checks)
    write_json('RESULTS.json', {'sorted_scenes': scenes, 'per_seed': summaries, 'metric_means': means})
    assert all(sha(ROOT / p) == h for p, h in identities.items())
    write_json('RECEIPT.json', {'input_sha256': identities, 'script_sha256': sha(Path(__file__)),
                               'new_model_calls': 0, 'new_training': 0, 'new_fits': 0,
                               'new_bootstraps': 0, 'new_image_or_label_pixel_reads': 0,
                               'remote_connections': 0, 'local_seconds': time.perf_counter() - start,
                               'pooled_optimum_scope': 'two fixed heads; whole-scene hard choice only, current 16 labeled test scenes',
                               'ties': 'scene objective: Source; pooled objective: fewer adapted scenes then smaller sorted-scene bitmask',
                               'main_head': '236 val-selected W, LR=1e-4 alpha=.75; secondary actual C endpoint LR=1e-4'})
    print(json.dumps({'output': str(OUT), 'means': means, 'local_seconds': time.perf_counter() - start}))


if __name__ == '__main__':
    main()
