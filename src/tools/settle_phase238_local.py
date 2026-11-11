"""Read sealed Phase234--237 tables only; no model/array/dependency/network access."""
from pathlib import Path
import csv
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase238/phase238_20261028'
INPUTS = {}
CHECKS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(phase, name):
    date = '20261027' if phase < 237 else '20261028'
    p = ROOT / f'outputs/phase{phase}/phase{phase}_{date}' / name
    INPUTS[p] = {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                 'sha256': sha(p)}
    return json.loads(p.read_text(encoding='utf-8-sig'))


def check(name, passed):
    if not passed:
        raise AssertionError(name)
    CHECKS.append({'check': name, 'passed': True})


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def miou(cm):
    values = []
    for i in range(3):
        u = sum(cm[i]) + sum(row[i] for row in cm) - cm[i][i]
        if u:
            values.append(cm[i][i] / u * 100)
    return sum(values) / len(values) if values else None


def mean(values):
    return sum(values) / len(values)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    tables = {p: load(p, 'TABLE2.json') for p in [234, 235, 236, 237]}
    comparisons = []
    for p, file in [(234, 'BOOTSTRAP_COMPARISONS.json'), (235, 'TABLE3.json'),
                    (236, 'TABLE3.json'), (237, 'TABLE3.json')]:
        data = load(p, file)
        comparisons.append({'phase': p, 'source_file': file, 'rows': data})
    for phase in [234, 235, 236, 237]:
        acceptance = load(phase, 'INDEPENDENT_ACCEPTANCE.json')
        check(f'phase{phase} prior acceptance status', acceptance['status'] == 'passed')
        date = '20261027' if phase < 237 else '20261028'
        p = ROOT / f'outputs/phase{phase}/phase{phase}_{date}/RESULT.md'
        INPUTS[p] = {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                     'sha256': sha(p)}

    # Recompute scalar pooled mIoU from existing table confusion, not from pixels.
    for phase in [235, 236, 237]:
        for i, row in enumerate(tables[phase]):
            cm = row.get('confusion')
            if cm is None:
                continue
            value = row.get('mIoU', row.get('pooled', {}).get('mIoU'))
            check(f'{phase} row{i} nonnegative integer confusion',
                  all(v >= 0 and int(v) == v for a in cm for v in a))
            check(f'{phase} row{i} mIoU recompute',
                  math.isclose(miou(cm), value, abs_tol=1e-8))

    metrics = ['mIoU', 'scene_mean_mIoU', 'Cloud_recall', 'Surface_to_Cloud_FPR',
               'Shadow_recall', 'nonShadow_to_Shadow_FPR']
    selected = []
    for domain in ['L8', 'SPARCS']:
        source = next(x for x in tables[236]
                      if x['domain'] == domain and x['strategy'] == 'Source')
        for strategy in ['Source', 'Source+B', 'E', 'W', 'P']:
            if strategy == 'Source':
                rows = [source]
            elif strategy == 'Source+B':
                rows = [x for x in tables[235] if x['domain'] == domain
                        and x['split'] == 'test' and x['strategy'] == 'Source+B']
                check(f'{domain} one calibrated Source', len(rows) == 1)
            else:
                rows = [x for x in tables[236] if x['domain'] == domain
                        and x['strategy'] == strategy and x['seed'] in [21701, 21702]]
                check(f'{domain} {strategy} exactly two seeds', len(rows) == 2)
            def value(row, key):
                return row[key] if key in row else row['pooled'][key]
            item = {'domain': domain, 'strategy': strategy,
                    'phase': 235 if strategy == 'Source+B' else 236,
                    'aggregation': 'single Source' if len(rows) == 1 else 'mean of two seed metrics'}
            item.update({m: mean([value(x, m) for x in rows]) for m in metrics})
            item['delta_pooled_mIoU_vs_Source_pp'] = item['mIoU'] - source['mIoU']
            item['delta_scene_mean_mIoU_vs_Source_pp'] = item['scene_mean_mIoU'] - source['scene_mean_mIoU']
            selected.append(item)
    write_csv('MAIN_RESULTS.csv', selected)

    strata = load(236, 'L8_TEST_STRATA.json')
    strata_summary = []
    for stratum in ['FULL', 'yes', 'no']:
        source = next(x for x in strata if x['stratum'] == stratum and x['strategy'] == 'Source')
        for strategy in ['Source', 'E', 'W', 'P']:
            rows = [source] if strategy == 'Source' else [
                x for x in strata if x['stratum'] == stratum and x['strategy'] == strategy
                and x['seed'] in [21701, 21702]]
            check(f'L8 {stratum} {strategy} row count', len(rows) == (1 if strategy == 'Source' else 2))
            item = {'stratum': stratum, 'strategy': strategy,
                    'mIoU': mean([x['mIoU'] for x in rows]),
                    'scene_mean_mIoU': mean([x['scene_mean_mIoU'] for x in rows])}
            item['delta_pooled_mIoU_vs_Source_pp'] = item['mIoU'] - source['mIoU']
            strata_summary.append(item)
    write_csv('L8_STRATA.csv', strata_summary)

    subset_summary = []
    for subset in ['SUBSET_1', 'SUBSET_2', 'SUBSET_3']:
        for strategy in ['E', 'W', 'P', 'endpoint']:
            rows = [x for x in tables[237] if x['subset'] == subset and x['strategy'] == strategy]
            check(f'{subset} {strategy} two seeds', len(rows) == 2)
            check(f'{subset} {strategy} both test gains positive',
                  all(x['test_mIoU_minus_Source_pp'] > 0 for x in rows))
            subset_summary.append({'subset': subset, 'strategy': strategy,
                                   'test_mIoU': mean([x['mIoU'] for x in rows]),
                                   'test_mIoU_minus_Source_pp': mean([
                                       x['test_mIoU_minus_Source_pp'] for x in rows]),
                                   'sign_flips_val_positive_test_negative': sum(
                                       bool(x['val_positive_test_negative']) for x in rows)})
    write_csv('SUBSET_RESULTS.csv', subset_summary)
    cost_rows = []
    for phase in [234, 235, 236, 237]:
        c = load(phase, 'COST_AND_IDENTITY_RECEIPT.json')
        call_key = {234: 'calls', 235: 'total_calls', 236: 'total_full_input_forwards',
                    237: 'total_input_forwards'}[phase]
        gpu_key = {234: 'GPU_stage_seconds', 235: 'GPU_stage_seconds_cumulative',
                   236: 'GPU_stage_seconds_total', 237: 'GPU_stage_seconds_total'}[phase]
        cost_rows.append({'phase': phase, 'input_forwards_including_failures': c[call_key],
                          'GPU_stage_seconds_including_failures': c[gpu_key],
                          'optimizer_updates_including_failures': c.get('total_optimizer_updates', c.get('optimizer_updates', 0)),
                          'successful_training_fits': {234: 0, 235: 0, 236: 8, 237: 6}[phase],
                          'new_scalar_fits': 1 if phase == 235 else 0})
    write_csv('COST_LEDGER.csv', cost_rows)
    totals = {'input_forwards': sum(x['input_forwards_including_failures'] for x in cost_rows),
              'GPU_stage_seconds': sum(x['GPU_stage_seconds_including_failures'] for x in cost_rows),
              'optimizer_updates': sum(x['optimizer_updates_including_failures'] for x in cost_rows),
              'successful_training_fits': 14, 'successful_fit_updates': 56000,
              'failed_updates_retained': 63, 'new_scalar_fits': 1,
              'scope': '234--237 incremental measured stages only; excludes Source pretraining, earlier phases, money and deployment latency'}
    check('cost update sum', totals['optimizer_updates'] == 56063)
    check('forward sum', totals['input_forwards'] == 133160)
    # Originals must remain byte-identical throughout this local settlement.
    for path, identity in INPUTS.items():
        check(f'input unchanged: {identity["path"]}', sha(path) == identity['sha256'])
    write_json('COPIED_COMPARISONS.json', comparisons)
    write_json('COST_TOTALS.json', totals)
    write_json('EVIDENCE_INDEX.json', list(INPUTS.values()))
    write_json('LOCAL_CHECKS.json', {'status': 'passed', 'checks': CHECKS,
                                   'scope': 'local table arithmetic, prior acceptance receipt status, original input hashes unchanged; no replay of model/pixel/bootstrap execution'})

    lines = ['# Phase238 汇总数表', '',
             '全部由234—237现有计数／指标整理；没有新预测、拟合或bootstrap。均值是指标均值，不是模型集成。', '',
             '| 目标 | 策略 | test池化mIoU | test景均mIoU | 池化相对Source |',
             '|---|---|---:|---:|---:|']
    for row in selected:
        lines.append(f"| {row['domain']} | {row['strategy']} | {row['mIoU']:.3f} | {row['scene_mean_mIoU']:.3f} | {row['delta_pooled_mIoU_vs_Source_pp']:+.3f} |")
    lines += ['', 'Source+B来自235；E/W/P来自236。L8支持为VALID，SPARCS原支持。不同方法监督／选模预算不全相同，主表不是同预算新排名。', '',
              '| L8评价层 | E−Source | W−Source | P−Source |', '|---|---:|---:|---:|']
    for st in ['FULL', 'yes', 'no']:
        vals = [next(x['delta_pooled_mIoU_vs_Source_pp'] for x in strata_summary
                     if x['stratum'] == st and x['strategy'] == strategy) for strategy in ['E', 'W', 'P']]
        lines.append(f'| {st} | ' + ' | '.join(f'{v:+.3f}' for v in vals) + ' |')
    lines += ['', 'yes是有云影标注六景；no十景不等于真实无云影。分层仅诊断，不替换完整16景主分析。', '',
              '| SPARCS子集 | E test | W test | P test |', '|---|---:|---:|---:|']
    for st in ['SUBSET_1', 'SUBSET_2', 'SUBSET_3']:
        vals = [next(x['test_mIoU'] for x in subset_summary if x['subset'] == st and x['strategy'] == strategy) for strategy in ['E', 'W', 'P']]
        lines.append(f'| {st} | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    lines += ['', f"234—237增量成本：{totals['input_forwards']:,}次前向，{totals['optimizer_updates']:,}次更新（含63次失败更新），14次成功训练、一次新标量拟合；累计GPU阶段{totals['GPU_stage_seconds']/3600:.3f}小时。不是整个项目总成本，也不是纯GPU内核时间。", '']
    (OUT / 'DATA_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json('LOCAL_RUN_RECEIPT.json', {'mode': 'local CPU, standard-library table processing',
                                       'script_path': str(Path(__file__)), 'script_sha256': sha(Path(__file__)),
                                       'new_training_updates': 0, 'new_inference': 0,
                                       'new_fits': 0, 'new_bootstraps': 0, 'remote_connections': 0,
                                       'image_label_pixel_reads': 0,
                                       'local_elapsed_seconds': time.perf_counter() - start})
    print(json.dumps({'output': str(OUT), 'checks': len(CHECKS), 'costs': totals}))


if __name__ == '__main__':
    main()
