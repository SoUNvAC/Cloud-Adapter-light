"""Produce the single CAL-1 delivery table from sealed Phase122 evidence."""
import csv
import json
import hashlib
from pathlib import Path

R = Path(__file__).resolve().parents[2]
U = R / 'outputs/phase122/phase122_20261015'
O = R / 'src/research_results/phase122_20261015'


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    with p.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fmt(x):
    return 'NA' if x is None or x == '' else f'{float(x):.4f}'


def main():
    d = U / 'delivery'
    controller = load(d / 'CONTROLLER_STATUS.json')
    assert controller['status'] == 'completed_pending_local_reference_join'
    train = load(d / 'train_C/TRAINING_RECEIPT.json')
    assert train['seed'] == 12201 and train['steps'] == 1000
    assert train['encoder_unchanged'] and train['decoder_updated']
    assert train['trainable_parameters'] == 14135988
    assert not train['teacher_state_sha256']
    assert sha(U / 'C_decoder_final.safetensors') == train['final_model_sha256']
    assert load(O / 'LOCAL_CPU_ACCEPTANCE.json')['status'] == 'passed'
    assert load(O / 'CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json')['status'] == 'passed'
    costs = [load(d / stage / 'RUN_STATUS.json') for stage in
             ('train_C', 'R_Calibration', 'C_Calibration', 'C_Catalogue', 'C_PixBox')]
    calls = sum(s['all_calls'] for s in costs)
    assert calls == controller['calls_completed'] <= 3500
    assert controller['elapsed_seconds'] <= 7200
    assert all(s['rss_peak_bytes'] <= 16 * 1024**3 and
               s['gpu_reserved_peak_bytes'] <= 20 * 1024**3 and
               s['host_min_free_bytes'] >= 4000000000 and
               s['intermediate_peak_bytes'] <= 4000000000 for s in costs)
    thresholds = load(d / 'THRESHOLD_LOCK.json')['thresholds']
    text = ['|数据池|模型／工作点|阈值|AP有效组／误报有效组|宏AP %|宏召回 %|平均负类误报 %|误报U95 %|最坏组及误报 %|备注|',
            '|---|---|---|---|---|---|---|---|---|---|']
    for pool in ('Catalogue', 'PixBox'):
        summary = rows(O / (pool + '_four_workpoints.csv'))
        per = rows(O / (pool + '_per_group.csv'))
        cmp = load(O / (pool + '_comparisons.json'))['rows']
        acceptance = load(O / (pool + '_CPU_ACCEPTANCE.json'))
        assert acceptance['status'] == 'passed'
        assert len(summary) == 4 and len(cmp) == 3
        assert {x['group_id'] for x in per}.__len__() == (32 if pool == 'Catalogue' else 10)
        for arm in ('R', 'C100'):
            a, b = [v for v in summary if v['setting'].startswith(arm + '_')]
            assert a['macro_AP_pp'] == b['macro_AP_pp'] and a['pooled_AP_pp'] == b['pooled_AP_pp']
        for v in summary:
            setting = v['setting']
            worst = max((r for r in per if r['setting'] == setting and r['FPR'] != ''), key=lambda r: float(r['FPR']))
            products = worst['products'].split(',')
            tiles = ','.join(sorted({p.split('_T')[1][:5] for p in products}))
            note = [f"池化AP {fmt(v['pooled_AP_pp'])}%；精确率 {fmt(v['macro_precision_pp'])}%；Surface/Cloud误报 {fmt(v['macro_Surface_FPR_pp'])}/{fmt(v['macro_Cloud_FPR_pp'])}%",
                    f"超1% {v['over_1pct_FP_groups']}组；零召回 {v['zero_recall_positive_groups']}组；共同支持损失 {acceptance['support_loss']}"]
            if pool == 'PixBox':
                note.append('采样点误报')
            if setting == 'C100_cal':
                c = next(c for c in cmp if c['comparison'] == 'C100_cal-R_cal')
                ci = c['recall']['CI95']
                note.append(f"较R/T_cal召回差 {fmt(c['recall']['mean'])} pp，95%描述区间 [{fmt(ci[0])}, {fmt(ci[1])}]；误报差 {fmt(c['FPR']['mean'])} pp")
                damage = min((r for r in rows(O / (pool + '_group_differences.csv')) if r['comparison'] == 'C100_cal-R_cal' and r['recall_delta_pp'] != ''), key=lambda r: float(r['recall_delta_pp']))
                note.append(f"最差组召回差 {fmt(damage['recall_delta_pp'])} pp（{damage['group_id']}）")
            arm = 'C' if setting.startswith('C100') else 'R'
            threshold = thresholds[arm] if setting.endswith('_cal') else 0.5
            values = [pool, setting.replace('_05', '/0.5').replace('_cal', '/T_cal'), repr(threshold),
                      v['AP_groups'] + '/' + v['negative_groups'], fmt(v['macro_AP_pp']), fmt(v['macro_recall_pp']),
                      fmt(v['macro_FPR_pp']), fmt(v['FPR_U95_pp']), tiles + ' / ' + fmt(v['max_group_FPR_pp']), '；'.join(note)]
            text.append('|' + '|'.join(values) + '|')
    elapsed = controller['elapsed_seconds'] + load(O / 'LOCAL_CPU_ACCEPTANCE.json')['elapsed_seconds']
    text.append('|运行状态|完成|—|—|—|—|—|—|—|' +
                f"控制器 {controller['elapsed_seconds']:.1f}s；本地评价后合计 {elapsed:.1f}s（不含传输／接手等待）；{calls}次调用；锁、全量校准边界、支持与CPU检查通过；原作业接续，无新增训练；执行端 /home/scv/shared/phase122_20261015；完整结果本地备份见 remote_backup|" )
    target = U / 'phasecal-1-result.md'
    with target.open('x', encoding='utf-8') as f:
        f.write('\n'.join(text) + '\n')
    print(target)


if __name__ == '__main__':
    main()
