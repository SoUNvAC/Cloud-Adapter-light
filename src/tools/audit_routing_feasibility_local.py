"""Development-support gate, SPARCS oracle, fixed cloud-fraction diagnostic."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/research_audits/routing_feasibility_20261028'
INPUT = ROOT / 'outputs/phase236/phase236_20261027/PER_SCENE.json'
MANIFEST = ROOT / 'src/research_plans/phase236_frozen_20261027/EVAL_INPUTS.json'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def flat(r):
    return [x for a in r['confusion'] for x in a]


def aggregate(rows):
    return [sum(x[j] for x in rows) for j in range(9)]


def miou(a):
    values = []
    for k in range(3):
        u = sum(a[k * 3:k * 3 + 3]) + sum(a[j * 3 + k] for j in range(3)) - a[k * 4]
        if u:
            values.append(a[k * 4] / u * 100)
    return sum(values) / len(values)


def ranks(values):
    out = [0.] * len(values)
    ix = sorted(range(len(values)), key=lambda i: values[i])
    i = 0
    while i < len(ix):
        j = i + 1
        while j < len(ix) and values[ix[j]] == values[ix[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            out[ix[k]] = rank
        i = j
    return out


def spearman(xs, ys):
    x, y = ranks(xs), ranks(ys)
    mx, my = statistics.mean(x), statistics.mean(y)
    den = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den if den else None


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in [INPUT, MANIFEST]}
    rows = json.loads(INPUT.read_text(encoding='utf-8-sig'))
    manifests = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    support, diagnostics, correlations, sparcs = [], [], [], []
    for domain in ['L8', 'SPARCS']:
        for split in ['val', 'test']:
            pool = [x for x in rows if x['domain'] == domain and x['split'] == split]
            source = {}
            for row in pool:
                if row['source']:
                    if row['scene'] in source:
                        assert source[row['scene']]['confusion'] == row['confusion']
                    source[row['scene']] = row
            meta = {}
            for row in manifests[domain][split]:
                meta.setdefault(row['scene'], row)
            for seed in [21701, 21702]:
                chosen = {x['scene']: x for x in pool if x['strategy'] == 'W'
                          and x['seed'] == seed and not x['source'] and x['lr'] == 1e-4
                          and x['alpha'] == (.75 if domain == 'L8' else 1.)}
                assert set(source) == set(chosen)
                scenes = sorted(source)
                a = [flat(source[s]) for s in scenes]
                b = [flat(chosen[s]) for s in scenes]
                gain = [miou(bb) - miou(aa) for aa, bb in zip(a, b)]
                source_n = sum(x < 0 for x in gain)
                support.append({'domain': domain, 'split': split, 'seed': seed,
                                'scenes': len(scenes), 'Source_better': source_n,
                                'W_better': sum(x > 0 for x in gain), 'ties': sum(x == 0 for x in gain),
                                'Source_better_scenes': json.dumps([s for s, g in zip(scenes, gain) if g < 0])})
                xs = []
                for i, s in enumerate(scenes):
                    aa, bb = a[i], b[i]
                    assert [sum(aa[k:k + 3]) for k in [0, 3, 6]] == [sum(bb[k:k + 3]) for k in [0, 3, 6]]
                    size = sum(aa)
                    source_cloud = sum(aa[k * 3 + 1] for k in range(3)) / size * 100
                    xs.append(source_cloud)
                    diagnostics.append({'domain': domain, 'split': split, 'seed': seed, 'scene': s,
                                        'biome_metadata': meta[s].get('biome', 'not provided'),
                                        'Shadows_annotation_metadata': meta[s].get('usgs_shadows', 'not provided'),
                                        'valid_reference_pixels': size,
                                        'Source_predicted_Cloud_pct': source_cloud,
                                        'reference_Cloud_pct_LABEL_DEPENDENT': sum(aa[3:6]) / size * 100,
                                        'Source_mIoU_LABEL_DEPENDENT': miou(aa),
                                        'W_mIoU_LABEL_DEPENDENT': miou(bb),
                                        'gain_W_minus_Source_pp': gain[i], 'prefer_Source_LABEL_DEPENDENT': gain[i] < 0})
                correlations.append({'domain': domain, 'split': split, 'seed': seed, 'scenes': len(scenes),
                                     'fixed_signal': 'Source predicted Cloud fraction on existing valid support',
                                     'Spearman_signal_vs_gain': spearman(xs, gain),
                                     'scope': 'descriptive only; no test tuning, no p-value, no deployable rule'})
                if domain != 'SPARCS':
                    continue
                scene_mask = sum(1 << i for i, g in enumerate(gain) if g > 0)
                def cm(mask):
                    return aggregate([b[i] if mask & (1 << i) else a[i] for i in range(len(scenes))])
                values = [miou(cm(mask)) for mask in range(1 << len(scenes))]
                best = min(range(len(values)), key=lambda mask: (-values[mask], mask.bit_count(), mask))
                assert values[best] >= max(miou(aggregate(a)), miou(aggregate(b))) - 1e-9
                sparcs.append({'split': split, 'seed': seed, 'scenes': len(scenes),
                               'all_Source_pooled': miou(aggregate(a)), 'all_W_pooled': miou(aggregate(b)),
                               'scene_oracle_pooled': miou(cm(scene_mask)),
                               'scene_oracle_macro': statistics.mean(max(miou(x), miou(y)) for x, y in zip(a, b)),
                               'scene_oracle_Source_count_including_ties': len(scenes) - scene_mask.bit_count(),
                               'exact_pooled_oracle': values[best],
                               'exact_pooled_oracle_gain_over_W_pp': values[best] - miou(aggregate(b)),
                               'exact_pooled_oracle_Source_count': len(scenes) - best.bit_count(),
                               'exact_pooled_oracle_Source_scenes': json.dumps([s for i, s in enumerate(scenes) if not best & (1 << i)]),
                               'routes_enumerated': len(values)})
    write_csv('ORACLE_PREFERENCE_COUNTS.csv', support)
    write_csv('SPARCS_ORACLES.csv', sparcs)
    write_csv('SCENE_ATTRIBUTES.csv', diagnostics)
    write_csv('FIXED_CLOUD_SIGNAL_CORRELATIONS.csv', correlations)
    assert all(digest(ROOT / p) == h for p, h in hashes.items())
    write_json('RECEIPT.json', {'inputs_sha256': hashes, 'script_sha256': digest(Path(__file__)),
                               'L8_val_development_gate': 'one unique Source-preferred scene for both seeds; no threshold/soft-route fitting',
                               'new_training': 0, 'new_model_calls': 0, 'new_fits': 0, 'remote_connections': 0,
                               'new_bootstraps': 0, 'new_image_or_label_pixel_reads': 0,
                               'local_seconds': time.perf_counter() - start})
    print(json.dumps({'support': support, 'SPARCS': sparcs, 'correlations': correlations}))


if __name__ == '__main__':
    main()
