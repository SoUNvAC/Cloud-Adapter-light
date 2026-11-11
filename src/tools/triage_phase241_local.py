"""Verify Phase241 delivery and decompose pooled pairwise AUC using tables only."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'outputs/phase241/phase241_20261030/delivery'
OUT = ROOT / 'outputs/research_audits/phase241_triage_20261031'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                      allow_nan=False) + '\n', encoding='utf-8')


def table(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(SRC / 'SHA256SUMS.json')
    for name, value in manifest.items():
        assert sha(SRC / name) == value, name
    receipt = read(SRC / 'INDEPENDENT_ACCEPTANCE.json')
    assert receipt['passed']
    with (SRC / 'RANKING.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    parts, scene_rows = [], []
    for domain in ('L8', 'SPARCS'):
        reference = None
        for model in ('Source', 'C21701', 'C21702'):
            subset = [r for r in rows if r['domain'] == domain and r['model'] == model
                      and r['split'] == 'test' and r['state'] == 'raw'
                      and r['support'] == 'main' and r['task'] == 'Shadow_Surface']
            scenes = [r for r in subset if r['scope'] == 'scene']
            assert len(scenes) == (6 if domain == 'L8' else 10)
            support = {r['scene']: (int(r['positive']), int(r['negative'])) for r in scenes}
            assert all(p > 0 and n > 0 for p, n in support.values())
            if reference is None:
                reference = support
            else:
                assert reference == support
            npos = sum(p for p, _ in support.values())
            nneg = sum(n for _, n in support.values())
            within_pairs = sum(p * n for p, n in support.values())
            all_pairs = npos * nneg
            within_weight = within_pairs / all_pairs
            within_auc = sum(int(r['positive']) * int(r['negative']) * float(r['AUC'])
                             for r in scenes) / within_pairs
            pooled = next(r for r in subset if r['scope'] == 'pooled')
            assert int(pooled['positive']) == npos and int(pooled['negative']) == nneg
            pooled_auc = float(pooled['AUC'])
            cross_auc = (pooled_auc - within_weight * within_auc) / (1 - within_weight)
            assert 0 <= cross_auc <= 1
            assert abs(within_weight * within_auc + (1 - within_weight) * cross_auc - pooled_auc) < 1e-12
            macro = sum(float(r['AUC']) for r in scenes) / len(scenes)
            declared = float(next(r for r in subset if r['scope'] == 'scene_mean')['AUC'])
            assert abs(macro - declared) < 1e-12
            parts.append(dict(domain=domain, model=model, scenes=len(scenes),
                              within_pair_weight=within_weight,
                              between_pair_weight=1-within_weight,
                              scene_mean_AUC=macro, within_pair_weighted_AUC=within_auc,
                              between_scene_AUC_inferred=cross_auc, pooled_AUC=pooled_auc))
            for r in scenes:
                p, n = support[r['scene']]
                scene_rows.append(dict(domain=domain, model=model, scene=r['scene'],
                                       positive=p, negative=n, AUC=float(r['AUC']),
                                       within_pair_weight_normalized=p*n/within_pairs))
    contrasts = []
    for domain in ('L8', 'SPARCS'):
        base = next(r for r in parts if r['domain'] == domain and r['model'] == 'Source')
        for model in ('C21701', 'C21702'):
            c = next(r for r in parts if r['domain'] == domain and r['model'] == model)
            a = base['within_pair_weight']
            within_delta = c['within_pair_weighted_AUC'] - base['within_pair_weighted_AUC']
            between_delta = c['between_scene_AUC_inferred'] - base['between_scene_AUC_inferred']
            total_delta = c['pooled_AUC'] - base['pooled_AUC']
            assert abs(a*within_delta + (1-a)*between_delta - total_delta) < 1e-12
            contrasts.append(dict(domain=domain, model=model,
                macro_delta_pp=100*(c['scene_mean_AUC']-base['scene_mean_AUC']),
                within_weighted_delta_pp=100*within_delta,
                between_scene_delta_pp=100*between_delta,
                within_contribution_pp=100*a*within_delta,
                between_contribution_pp=100*(1-a)*between_delta,
                pooled_delta_pp=100*total_delta))
    table('PAIR_AUC_DECOMPOSITION.csv', parts)
    table('PAIR_AUC_CONTRASTS.csv', contrasts)
    table('SCENE_PAIR_SUPPORT.csv', scene_rows)
    save('RECEIPT.json', dict(status='passed', sealed_delivery_files_verified=len(manifest),
        reported_independent_checks=receipt['checks'],
        decomposition='pooled AUC = alpha * pair-weighted within-scene AUC + (1-alpha) * between-scene AUC',
        post_hoc=True, new_inference=0, new_fits=0, new_bootstrap=0,
        no_remote_connection=True, no_score_or_image_payload_read=True,
        input_SHA256={str((SRC / name).relative_to(ROOT)):sha(SRC / name)
                      for name in ('RANKING.csv','SHA256SUMS.json','INDEPENDENT_ACCEPTANCE.json')},
        code_SHA256=sha(Path(__file__))))
    save('SHA256SUMS.json', {p.name:sha(p) for p in sorted(OUT.iterdir())
                           if p.is_file() and p.name != 'SHA256SUMS.json'})
    print(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
