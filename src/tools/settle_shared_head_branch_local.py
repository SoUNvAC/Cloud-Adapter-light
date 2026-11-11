"""Reorganize frozen metrics only. No inference, selection, CI, or remote IO."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/research_audits/shared_head_settlement_20261026'
INPUTS = {}


def read_bytes(relative):
    path = ROOT / relative
    data = path.read_bytes()
    INPUTS[relative] = hashlib.sha256(data).hexdigest()
    return data


def rows(relative):
    return list(csv.DictReader(read_bytes(relative).decode('utf-8-sig').splitlines()))


def save(name, data):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = 'outputs/phase227/phase227_20261025/'
    candidates = rows(prefix + 'TABLE_B_ALL_CANDIDATES.csv')
    rules = rows(prefix + 'SOURCE_RULE_ALL_ARMS.csv')
    costs = json.loads(read_bytes(prefix + 'HISTORICAL_TRAINING_COST.json'))
    read_bytes('outputs/phase228/phase228_20261026/RESULT.md')
    read_bytes('outputs/research_audits/head_routing_deployment_20261026/REVIEW.md')
    read_bytes('outputs/research_audits/head_routing_deployment_20261026/STORAGE_AND_ROUTING_SCOPE.csv')
    index = {(r['domain'], r['seed'], r['arm'], float(r['alpha'])): r for r in candidates}
    assert len(index) == len(candidates) == 100
    locked = {(r['domain'], r['seed'], r['arm']): float(r['alpha'])
              for r in rules if float(r['budget_pp']) == 1}
    assert len(locked) == 20
    metrics = ['test_mIoU', 'pool_loss_pp', 'T11_pp', 'worst_ROI_loss_pp']
    per_seed = []
    for domain in ['L8', 'SPARCS']:
        for seed in ['21701', '21702']:
            for arm in ['C', 'D', 'H', 'W1', 'W10']:
                for policy, alpha in [('fixed_025', .25), ('endpoint_1', 1.),
                                      ('historical_B1_T11_5', locked[domain, seed, arm])]:
                    original = index[domain, seed, arm, alpha]
                    if policy == 'historical_B1_T11_5':
                        assert float(original['pool_loss_pp']) <= 1 + 1e-10
                        assert float(original['T11_pp']) <= 5 + 1e-10
                    record = dict(domain=domain, seed=seed, arm=arm, policy=policy,
                                  alpha=alpha, deployment='shared_head',
                                  extra_FP32_head_MiB=0.)
                    record.update({k: float(original[k]) for k in metrics})
                    per_seed.append(record)
                    routed = dict(record, deployment='conditional_Source_plus_same_target_head',
                                  extra_FP32_head_MiB=1316516 * 4 / 1024 ** 2)
                    routed.update(pool_loss_pp=0., T11_pp=0., worst_ROI_loss_pp=0.)
                    per_seed.append(routed)
    groups = defaultdict(list)
    for r in per_seed:
        groups[r['domain'], r['arm'], r['policy'], r['deployment']].append(r)
    means = []
    for key, group in groups.items():
        assert len(group) == 2
        record = dict(zip(['domain', 'arm', 'policy', 'deployment'], key))
        record['seed_alphas'] = '/'.join(str(r['alpha']) for r in group)
        record.update({k: mean(r[k] for r in group) for k in metrics})
        record['extra_FP32_head_MiB'] = group[0]['extra_FP32_head_MiB']
        means.append(record)
    cost_rows = []
    for key, arms in costs.items():
        domain, seed = key.split('/')
        for arm, values in arms.items():
            cost_rows.append(dict(domain=domain, seed=seed, arm=arm,
                                  training_units=values['training_units'],
                                  recorded_train_seconds=values['recorded_train_seconds'],
                                  source_replay=values['source_replay'],
                                  phase=values['phase']))
    assert len(cost_rows) == 12
    save('PER_SEED_STRATEGIES.csv', per_seed)
    save('MEAN_STRATEGIES.csv', means)
    save('HISTORICAL_CDH_TRAINING_COST.csv', cost_rows)
    audit = dict(inputs=INPUTS, source_table_rows=len(candidates),
                 per_seed_rows=len(per_seed), mean_rows=len(means),
                 policy='Reuse original B1/T11<=5 alpha, no new choice. Fixed alpha .25/1 retained.',
                 routing='Conditional identity construction, not new deployment measurement. Each target is a separate two-domain case.',
                 statistics='Arithmetic mean of two seed metrics, not score ensemble. No new intervals.',
                 model_calls=0, training_updates=0, remote_connections=0,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([r for r in means if r['deployment']=='shared_head'], indent=2))


if __name__ == '__main__':
    main()
