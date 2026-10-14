"""Independent CPU acceptance and fixed development-only screening."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path('/home/scv/shared/phase117_20261014')


def load(p):
    return json.loads(p.read_text())


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    state = load(ROOT / 'RUN_STATUS.json')
    assert state['status'] == 'training_and_development_complete_pending_cpu_acceptance'
    assert (ROOT / 'EXIT_CODE').read_text().strip() == '0'
    assert state['calls'] <= 9500 and state['training_calls'] == 8000
    assert state['elapsed_seconds'] <= 21600
    assert state['rss_peak'] <= 16*1024**3 and state['gpu_reserved_peak'] <= 20*1024**3
    assert load(ROOT / 'FINAL_BASE_UNCHANGED.json')['bitwise_unchanged']
    seal = load(ROOT / 'FOUR_TRAINING_SEAL.json')
    assert all(sha(ROOT / f'{n}_final.pt') == h for n, h in seal['models'].items())
    original = load(ROOT / 'DEVELOPMENT_GROUP_RESULTS.json')
    summaries, pooled, checked = {}, {}, 0
    group_table = []
    common_labels = {}
    for name, rows in original.items():
        scores_list, labels_list = [], []
        cm_total = np.zeros((3, 3), np.int64)
        for row in rows:
            cache = np.load(ROOT / 'scores' / f"{name}_{row['product']}.npz", allow_pickle=False)
            score, truth, pred = cache['scores'], cache['truth'], cache['prediction']
            digest = hashlib.sha256(truth.tobytes()).hexdigest()
            if row['product'] in common_labels:
                assert digest == common_labels[row['product']]
            common_labels[row['product']] = digest
            positive = truth == 2
            ap = float(average_precision_score(positive, score)) if positive.any() and (~positive).any() else None
            assert ap is None and row['ap'] is None or abs(ap-row['ap']) <= 1e-10
            # Independent confusion calculation, not the worker's bincount helper.
            matrix = np.array([[( (truth == a) & (pred == b)).sum() for b in range(3)] for a in range(3)], np.int64)
            assert matrix.tolist() == row['confusion']
            assert len(score) == row['positive'] + row['negative']
            scores_list.append(score)
            labels_list.append(truth)
            cm_total += matrix
            group_table.append({'policy': name, 'product': row['product'], 'group_id': row['group_id'],
                                'AP_pp': ap * 100 if ap is not None else None,
                                'recall_pp': row['recall'] * 100 if row['recall'] is not None else None,
                                'mIoU_pp': row['miou_percent'], 'positive': row['positive'], 'negative': row['negative']})
            checked += 1
        scores, labels = np.concatenate(scores_list), np.concatenate(labels_list)
        pooled_ap = float(average_precision_score(labels == 2, scores))
        union = cm_total.sum(0) + cm_total.sum(1) - np.diag(cm_total)
        ious = np.divide(np.diag(cm_total), union, out=np.full(3, np.nan), where=union > 0)
        ap_groups = [r for r in rows if r['ap'] is not None]
        assert len(ap_groups) == 7 and len(rows) == 32
        summaries[name] = {'policy': name, 'macro_AP_pp': float(np.mean([r['ap'] for r in ap_groups])*100),
                           'pooled_AP_pp': pooled_ap*100, 'pooled_mIoU_pp': float(np.nanmean(ious)*100),
                           'macro_Shadow_recall_pp': float(np.mean([r['recall'] for r in rows if r['recall'] is not None])*100),
                           'pooled_IoU_pp': (ious*100).tolist(), 'confusion': cm_total.tolist(),
                           'AP_groups': 7, 'all_groups': 32}
        pooled[name] = pooled_ap
        del scores_list, labels_list, scores, labels
    ids = [r['group_id'] for r in original['O'] if r['ap'] is not None]
    by = {n: {r['group_id']: r for r in rows} for n, rows in original.items()}
    seed_rows, diffs = [], []
    for seed in (11701, 11702):
        s, r = f'S{seed}', f'R{seed}'
        delta = np.array([(by[s][g]['ap'] - by[r][g]['ap'])*100 for g in ids])
        diffs.append(delta)
        seed_rows.append({'seed': seed, 'mean_S_minus_R_AP_pp': float(delta.mean()),
                          'S_minus_O_macro_AP_pp': summaries[s]['macro_AP_pp']-summaries['O']['macro_AP_pp'],
                          'S_minus_O_pooled_mIoU_pp': summaries[s]['pooled_mIoU_pp']-summaries['O']['pooled_mIoU_pp'],
                          'S_minus_O_macro_Shadow_recall_pp': summaries[s]['macro_Shadow_recall_pp']-summaries['O']['macro_Shadow_recall_pp']})
    delta = (diffs[0]+diffs[1])/2
    rng = np.random.default_rng(117014)
    indexes = rng.integers(7, size=(10000, 7))
    estimates = delta[indexes].mean(1)
    ci = np.quantile(estimates, [.025, .975])
    # Independent scalar replication of the fixed pre-drawn bootstrap, no new resamples.
    scalar = np.array([sum(float(delta[i]) for i in ix)/7 for ix in indexes])
    assert np.max(np.abs(estimates-scalar)) <= 1e-10
    conditions = {'mean_S_R_at_least_1pp': float(delta.mean()) >= 1,
                  'paired_group_CI_lower_gt_0': float(ci[0]) > 0,
                  'each_seed_S_R_gt_0': all(r['mean_S_minus_R_AP_pp'] > 0 for r in seed_rows),
                  'each_seed_S_macro_AP_not_below_O': all(r['S_minus_O_macro_AP_pp'] >= 0 for r in seed_rows),
                  'each_seed_S_pooled_mIoU_loss_at_most_1pp': all(r['S_minus_O_pooled_mIoU_pp'] >= -1 for r in seed_rows),
                  'each_seed_S_macro_Shadow_recall_loss_at_most_1pp': all(r['S_minus_O_macro_Shadow_recall_pp'] >= -1 for r in seed_rows)}
    primary = {'comparison': 'S-R, seeds averaged within related group first', 'n_groups': 7,
               'mean_delta_AP_pp': float(delta.mean()), 'CI95_pp': ci.tolist(),
               'seed': 117014, 'resamples': 10000, 'conditions': conditions,
               'passed': all(conditions.values()), 'per_seed': seed_rows,
               'per_group': [{'group_id': g, 'seed11701_delta_pp': float(diffs[0][i]),
                              'seed11702_delta_pp': float(diffs[1][i]), 'mean_delta_pp': float(delta[i])} for i, g in enumerate(ids)],
               'unit': 'related group, seeds are not extra independent scenes',
               'exploratory_screening_not_noninferiority_or_independent_confirmation': True}
    write('PRIMARY_SCREENING.json', primary)
    write('FIVE_POLICY_SUMMARY.json', summaries)
    write('CPU_METRIC_ACCEPTANCE.json', {'status': 'passed', 'group_policy_checks': checked,
           'independent_sklearn_AP_and_confusion': True, 'common_label_SHA_across_all_policies': True,
           'bootstrap_scalar_replay_max_error': float(np.max(np.abs(estimates-scalar))),
           'CPU_new_model_calls': 0, 'all_frozen_base_and_cost_gates_passed': True})
    with (ROOT / 'per_group_per_seed.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(group_table[0])); w.writeheader(); w.writerows(group_table)
    table = '\n'.join(f"|{n}|{r['macro_AP_pp']:.3f}|{r['pooled_AP_pp']:.3f}|{r['pooled_mIoU_pp']:.3f}|{r['macro_Shadow_recall_pp']:.3f}|" for n,r in summaries.items())
    decision = '固定探索继续条件全部满足；仅允许另立稳定性／外部验证问题，不自动启动。' if primary['passed'] else '至少一项固定继续条件失败，关闭本次固定补丁方案，不追加步数、seed、宽度或强度。'
    report = f'''# Phase117固定输入适配探索

## 1．数据结果

仅Catalogue原153个fit代表拟合，32个development代表评价（7个有影相关组）。冻结Omni V1双底座，共享6→3输入1x1补丁21参数，零初始化；R复制原归一化RG/B8A，S追加归一化B02/B11/B12。seed11701／11702每种1000步，原滑窗/nodata调度与集成不变，父类按概率合并。原底座参数／buffer逐位不变，零补丁及梯度实现门通过。

|策略|宏AP(%)|池化AP(%)|池化三类mIoU(%)|宏Shadow召回(%)|
|---|---:|---:|---:|---:|
{table}

主要S−R先组内平均两个seed，再7组等权：{primary['mean_delta_AP_pp']:.3f} AP百分点，配对组bootstrap95%描述区间[{ci[0]:.3f},{ci[1]:.3f}]，10000次seed117014。每seed／组结果及六项继续条件逐项见PRIMARY_SCREENING.json，未将seed或像元当独立场景。实际网络调用{state['calls']}，累计墙钟{state['elapsed_seconds']:.1f}秒；训练8000调用，其余构建／门／评价如RUN_STATUS。CPU逐组AP/混淆及标量区间复核通过。

## 2．结论分析

{decision}
这是已暴露development上的可行性筛选，未知Omni训练交集和半自动参考语义限制保持；不能称首次性、机制实证、统计非劣或TGRS方法贡献。R／S名义参数相同但输入秩不同，不称函数容量严格相等。训练512裁剪与评价1000滑窗上下文不同；本数据发布算子不等于原生SAFE严格复现。旧G7及补充评分分枝继续关闭，未访问PixBox／ALCD／其他确认池，无新下载、调阈值或checkpoint选择。
'''
    (ROOT / 'README.md').write_text(report, encoding='utf-8')
    delivered = ['INPUT_GATE.json', 'IMPLEMENTATION_GATE.json', 'GRADIENT_GATE_R.json', 'GRADIENT_GATE_S.json',
                 'PAIRED_CROP_SCHEDULE.json', 'FROZEN_BASE_LOCK.json', 'FOUR_TRAINING_SEAL.json',
                 'FINAL_BASE_UNCHANGED.json', 'PRIMARY_SCREENING.json', 'FIVE_POLICY_SUMMARY.json',
                 'CPU_METRIC_ACCEPTANCE.json', 'per_group_per_seed.csv', 'RUN_STATUS.json', 'README.md']
    delivered += [f'{n}_training.json' for n in seal['models']]
    write('DELIVERY_SHA256.json', {n: sha(ROOT/n) for n in delivered})
    print(decision, flush=True)


if __name__ == '__main__':
    main()
