"""Produce the single CAL-1 delivery table from sealed Phase122 evidence."""
import csv
import json
import hashlib
import time
import struct
import sys
from pathlib import Path
import numpy as np

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


def write_table(text):
    target = U / 'phasecal-1-result.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(text) + '\n', encoding='utf-8')
    temporary.replace(target)
    print(target)


def main():
    headings = ['|数据池|模型／工作点|阈值|AP有效组／误报有效组|宏AP %|宏召回 %|平均负类误报 %|误报U95 %|最坏组及误报 %|备注|',
                '|---|---|---|---|---|---|---|---|---|---|']
    if not (O / 'LOCAL_CPU_ACCEPTANCE.json').exists():
        thresholds = load(U / 'THRESHOLD_LOCK.json')['thresholds']
        text = headings.copy()
        for pool in ('Catalogue', 'PixBox'):
            for arm in ('R', 'C100'):
                for cal in (False, True):
                    t = thresholds['R' if arm == 'R' else 'C'] if cal else .5
                    text.append(f"|{pool}|{arm}/{'T_cal' if cal else '0.5'}|{t!r}|NA|NA|NA|NA|NA|NA|等待分数传回本地及固定参考连接；未报告效果|" )
        text.append('|运行状态|部分完成|—|—|—|—|—|—|—|原控制器正常结束：2612调用、1361.1s；全量校准边界独立CPU复核通过；评价及完整备份传输中；执行端 /home/scv/shared/phase122_20261015；未重复训练，未新增模型调用|')
        write_table(text)
        return
    d = U / 'delivery'
    controller = load(d / 'CONTROLLER_STATUS.json')
    assert controller['status'] == 'completed_pending_local_reference_join'
    train = load(d / 'train_C/TRAINING_RECEIPT.json')
    assert train['seed'] == 12201 and train['steps'] == 1000
    assert train['encoder_unchanged'] and train['decoder_updated']
    assert train['trainable_parameters'] == 14135988
    assert not train['teacher_state_sha256']
    model = U / 'C_decoder_final.safetensors'
    assert sha(model) == train['final_model_sha256'] == load(d / 'TRAINING_SEAL.json')['models']['C']
    raw = model.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+size])
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    assert set(tensors) == set(load(d / 'train_C/PARAMETER_WHITELIST.json')['decoder_state_names'])
    end = 0
    for name, item in sorted(tensors.items(), key=lambda x: x[1]['data_offsets'][0]):
        a, b = item['data_offsets']
        assert a == end
        end = b
        tensor = np.frombuffer(raw[8+size+a:8+size+b], dtype={'F32': '<f4', 'I64': '<i8'}[item['dtype']])
        assert tensor.size == int(np.prod(item['shape'])) and np.isfinite(tensor).all(), name
    assert len(tensors) == 61 and end == len(raw)-8-size
    backup_verified = '--backup-verified' in sys.argv
    if backup_verified:
        assert (U / 'remote_backup/phase122_20261015/EXIT_CODE').read_text().strip() == '0'
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
    text = headings.copy()
    for pool in ('Catalogue', 'PixBox'):
        summary = rows(O / (pool + '_four_workpoints.csv'))
        per = rows(O / (pool + '_per_group.csv'))
        cmp = load(O / (pool + '_comparisons.json'))['rows']
        acceptance = load(O / (pool + '_CPU_ACCEPTANCE.json'))
        assert acceptance['status'] == 'passed'
        assert len(summary) == 4 and len(cmp) == 3
        assert len({x['group_id'] for x in per}) == (32 if pool == 'Catalogue' else 10)
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
    elapsed = time.time() - controller['started_epoch']
    stage_seconds = sum(s['elapsed_seconds'] for s in costs)
    status = '完成' if backup_verified else '部分完成'
    backup_note = '完整结果本地备份SHA已核对：remote_backup' if backup_verified else '评价完成；完整备份仍在传输，尚未验收'
    text.append(f'|运行状态|{status}|—|—|—|—|—|—|—|' +
                f"自原控制器启动至交付表生成共 {elapsed:.1f}s（含传输／接手等待）；控制器 {controller['elapsed_seconds']:.1f}s；模型阶段墙钟合计 {stage_seconds:.1f}s；{calls}次调用；锁、全量校准边界、支持与CPU检查通过；原作业接续，无新增训练；执行端 /home/scv/shared/phase122_20261015；{backup_note}|" )
    write_table(text)


if __name__ == '__main__':
    main()
