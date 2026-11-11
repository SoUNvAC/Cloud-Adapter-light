"""Compare existing V1 C100 and E; replay one already-frozen E threshold, no fits."""
import collections
import csv
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
O = R / 'outputs/research_audits/c100_vs_e_20261019'


def j(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_csv(path, data):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    O.mkdir(parents=True, exist_ok=True)
    refs_path = R / 'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    sealed_path = R / 'src/research_results/phase112_20261014/SCORES_SEALED.json'
    threshold_path = R / 'outputs/phase202/phase202_20261017/THRESHOLDS_FROZEN.json'
    score_paths = sorted((R / 'data/pixbox_s2_cmix_5036991/phase112_20261014/omni').glob('*/UNLABELLED_POINT_SCORES.json'))
    assert len(score_paths) == 16
    input_paths = [refs_path, sealed_path, threshold_path, Path(__file__)] + score_paths
    for pool in ('Catalogue', 'PixBox'):
        input_paths += [R / f'outputs/phase119/phase119_20261015/{pool}_summary.csv',
                        R / f'outputs/phase119/phase119_20261015/{pool}_per_group.csv',
                        R / f'src/research_results/phase124_20261017/{pool}_performance.csv',
                        R / f'outputs/phase123/phase123_20261016/{pool}_per_group.csv']
    input_paths += [R / 'outputs/phase202/phase202_20261017/Catalogue_SUMMARY.csv']
    (O / 'INPUT_CODE_LOCK.json').write_text(json.dumps(
        [{'path': str(p), 'sha256': sha(p)} for p in input_paths], indent=2), encoding='utf-8')
    sealed = j(sealed_path)['score_sha256']
    points = {}
    for path in score_paths:
        assert sha(path) == sealed[path.parent.name]
        for point in j(path)['points']:
            assert point['point_id'] not in points
            points[point['point_id']] = point
    assert len(points) == 9500
    truth = collections.defaultdict(list)
    label = {'positive_ground_cloud_shadow': 1, 'negative_surface': 0, 'negative_cloud': 0}
    for ref in j(refs_path):
        if ref['reference'] not in label:
            continue
        point = points[ref['point_ids'][0]]
        assert point['valid']
        truth[ref['group_id']].append((label[ref['reference']], point['shadow_confidence']))
    assert sum(y for data in truth.values() for y, _ in data) == 379
    assert sum(1 - y for data in truth.values() for y, _ in data) == 8190
    frozen_t = j(threshold_path)['thresholds']['E']['threshold']

    def count(t):
        result = {}
        for group, data in truth.items():
            pos = sum(y for y, _ in data)
            neg = len(data) - pos
            tp = sum(y == 1 and score >= t for y, score in data)
            fp = sum(y == 0 and score >= t for y, score in data)
            result[group] = dict(positive=pos, negative=neg, TP=tp, FP=fp,
                                 recall=tp / pos if pos else None, FPR=fp / neg if neg else None)
        return result

    counts05, counts_cal = count(.5), count(frozen_t)
    summary, comparisons, per_group = [], [], []
    for pool in ('Catalogue', 'PixBox'):
        original = next(x for x in rows(R / f'outputs/phase119/phase119_20261015/{pool}_summary.csv') if x['setting'] == 'E')
        eg = {x['group_id']: x for x in rows(R / f'outputs/phase119/phase119_20261015/{pool}_per_group.csv')}
        cg = rows(R / f'outputs/phase123/phase123_20261016/{pool}_per_group.csv')
        performance = rows(R / f'src/research_results/phase124_20261017/{pool}_performance.csv')
        if pool == 'Catalogue':
            e_cal = next(x for x in rows(R / 'outputs/phase202/phase202_20261017/Catalogue_SUMMARY.csv') if x['setting'] == 'E')
            e_recall, e_fpr = float(e_cal['macro_recall_pp']), float(e_cal['macro_FPR_pp'])
            assert int(e_cal['pooled_positive']) == int(original['pooled_positive'])
            assert int(e_cal['pooled_negative']) == int(original['pooled_negative'])
        else:
            for group, item in counts05.items():
                for key in ('positive', 'negative', 'TP', 'FP'):
                    assert item[key] == int(eg[group]['E_' + key])
            e_recall = sum(x['recall'] for x in counts_cal.values() if x['recall'] is not None) / 8 * 100
            e_fpr = sum(x['FPR'] for x in counts_cal.values()) / 10 * 100
        for setting in ('E_05', 'E_cal', 'C122_05', 'C122_cal', 'C123_05', 'C123_cal'):
            if setting.startswith('E'):
                item = dict(pool=pool, setting=setting, threshold=.5 if setting.endswith('05') else frozen_t,
                            macro_AP_pp=float(original['macro_AP_pp']), pooled_AP_pp=float(original['pooled_AP_pp']),
                            macro_recall_pp=float(original['macro_recall_pp']) if setting.endswith('05') else e_recall,
                            macro_FPR_pp=float(original['macro_negative_FP_pp']) if setting.endswith('05') else e_fpr,
                            positive=int(original['pooled_positive']), negative=int(original['pooled_negative']))
            else:
                old = next(x for x in performance if x['setting'] == setting)
                item = {key: float(old[key]) for key in ('threshold', 'macro_AP_pp', 'pooled_AP_pp', 'macro_recall_pp', 'macro_FPR_pp')}
                item.update(pool=pool, setting=setting, positive=int(old['pooled_positive']), negative=int(old['pooled_negative']))
            assert item['positive'] == int(original['pooled_positive']) and item['negative'] == int(original['pooled_negative'])
            summary.append(item)
        for arm in ('C122', 'C123'):
            for old in [x for x in cg if x['setting'] == arm + '_cal']:
                e = eg[old['group_id']]
                assert int(e['positive']) == int(old['positive']) and int(e['negative']) == int(old['negative'])
                assert e['products'] == old['products']
                per_group.append(dict(pool=pool, group_id=old['group_id'], products=old['products'], arm=arm,
                                      positive=int(old['positive']), negative=int(old['negative']),
                                      AP_delta_vs_E_pp=(float(old['AP']) - float(e['E_AP'])) * 100 if old['AP'] else ''))
            c = next(x for x in summary if x['pool'] == pool and x['setting'] == arm + '_cal')
            comparisons.append(dict(pool=pool, arm=arm, macro_AP_delta_vs_E_pp=c['macro_AP_pp'] - float(original['macro_AP_pp']),
                                    pooled_AP_delta_vs_E_pp=c['pooled_AP_pp'] - float(original['pooled_AP_pp']),
                                    cal_recall_delta_vs_E_pp=c['macro_recall_pp'] - e_recall,
                                    cal_FPR_delta_vs_E_pp=c['macro_FPR_pp'] - e_fpr))
    save_csv(O / 'WORKPOINTS.csv', summary)
    save_csv(O / 'C_MINUS_E.csv', comparisons)
    save_csv(O / 'PER_GROUP_AP_COMPARISON.csv', per_group)
    save_csv(O / 'PIXBOX_E_FROZEN_THRESHOLD_REPLAY.csv', [dict(group_id=g, **x) for g, x in counts_cal.items()])
    receipt = dict(status='passed', no_network_or_training_or_fits=True, no_new_threshold=True,
                   E_frozen_threshold=frozen_t, PixBox_E_cal_macro_recall_pp=e_recall,
                   note='PixBox scalar replay is in WORKPOINTS; no intervals recomputed',
                   paired_C_minus_E_intervals='not_computed', C_matched_deployment_timing='unmeasured')
    (O / 'ACCEPTANCE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'comparison': comparisons, 'workpoints': summary}, indent=2))


if __name__ == '__main__':
    main()
