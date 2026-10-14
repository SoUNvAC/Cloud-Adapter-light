"""Seal the user's final submission, then report post-hoc review without new pixels."""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'src/research_results/phase115_20261014'
LOCAL = ROOT / 'outputs/phase115/phase115_20261014'
PRIVATE = ROOT / 'data/pixbox_s2_cmix_5036991/phase115_20261014'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def present(value):
    return isinstance(value, str) and bool(value.strip()) and value != 'unknown'


def main():
    for name in ('DELIVERY_SHA256.json', 'DELIVERY_SHA256_V2.json'):
        manifest = json.loads((OUT / name).read_text(encoding='utf-8'))
        assert all(sha(ROOT / p) == h for p, h in manifest.items()), name
    source = LOCAL / 'reviewer_pack/independent_adjudication_submission.json'
    raw = source.read_bytes()
    submission = json.loads(raw)
    template = json.loads((LOCAL / 'reviewer_pack/independent_adjudication_template.json').read_text(encoding='utf-8'))
    rows = submission['records']
    assert [r['independent_id'] for r in rows] == [r['independent_id'] for r in template['records']]
    for row in rows:
        assert row['point_candidate_class'] in template['allowed_point_classes']
        assert row['cell_20m_state'] in template['allowed_cell_states']
        assert row['status'] in ('draft', 'reviewed', 'not_reviewed')
    sealed = PRIVATE / 'SEALED_PHASE115_FINAL_SUBMISSION.json'
    with sealed.open('xb') as f:
        f.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert sha(sealed) == digest
    write(PRIVATE / 'FINAL_SUBMISSION_RECEIPT.json', {
        'received_at': datetime.now(timezone.utc).isoformat(),
        'sha256': digest, 'bytes': len(raw),
        'acceptance': 'user explicitly requested completion; final with missing fields retained',
        'exported_at_as_provided': submission.get('actual_submitted_at'),
        'reviewer_metadata_as_provided': {k: submission.get(k) for k in
            ('reviewer_id', 'reviewer_experience', 'has_seen_original_reveal', 'identity_verification')},
        'context': 'same user has seen the Phase114 reveal; post-hoc review only',
        'raw_file_changed': False,
    })
    # Join controlled metadata only after the opinion bytes and receipt are sealed.
    lock = json.loads((OUT / 'SEVEN_CASE_ANON_LOCK.json').read_text(encoding='utf-8'))
    trace_path = PRIVATE / 'TRACE_PRIVATE.json'
    assert sha(trace_path) == lock['trace_sha256']
    trace = {r['independent_id']: r for r in json.loads(trace_path.read_text(encoding='utf-8'))['rows']}
    assert sha(ROOT / 'data/pixbox_s2_cmix_5036991/phase114_20261014/SEALED_BLIND_OPINIONS.json') == lock['original_opinion_sha256']
    assert all(sha(LOCAL / 'reviewer_pack/assets' / n) == h for n, h in lock['image_sha256'].items())
    joined = []
    for row in rows:
        meta = trace[row['independent_id']]
        tile = re.search(r'_T([0-9]{2}[A-Z]{3})_', meta['product_id']).group(1)
        complete = all(present(row.get(k)) for k in
            ('visible_evidence', 'uncertainty', 'existing_context_sufficient'))
        joined.append({'opinion': row, 'trace': meta, 'tile': tile, 'fields_complete': complete})
    write(PRIVATE / 'POSTHOC_REVIEW_JOINED_PRIVATE.json', {
        'sealed_submission_sha256': digest, 'classification': 'post_hoc_review_only', 'rows': joined})
    complete_groups = {r['trace']['group_id'] for r in joined if r['fields_complete']}
    summary = {
        'sealed_submission_sha256': digest, 'raw_bytes': len(raw), 'accepted_as_final': True,
        'classification': 'post_hoc_review_only', 'records': len(rows),
        'original_record_status_counts': dict(Counter(r['status'] for r in rows)),
        'candidate_class_counts': dict(Counter(r['point_candidate_class'] for r in rows)),
        'cell_state_counts': dict(Counter(r['cell_20m_state'] for r in rows)),
        'visible_evidence_provided': sum(present(r.get('visible_evidence')) for r in rows),
        'uncertainty_provided': sum(present(r.get('uncertainty')) for r in rows),
        'context_sufficient_self_report': sum(r.get('existing_context_sufficient') == '足够' for r in rows),
        'related_groups': len({r['trace']['group_id'] for r in joined}),
        'case_counts_by_tile': dict(Counter(r['tile'] for r in joined)),
        'complete_fields_cases': sum(r['fields_complete'] for r in joined),
        'complete_fields_groups': len(complete_groups),
        'complete_fields_tiles': sorted({r['tile'] for r in joined if r['fields_complete']}),
        'reviewer_metadata_as_provided': {k: submission.get(k) for k in
            ('reviewer_id', 'reviewer_experience', 'has_seen_original_reveal', 'identity_verification')},
        'actual_submitted_at_as_provided': submission.get('actual_submitted_at'),
        'eligible_independent_cases': 0, 'eligible_independent_groups': 0,
        'minimum_independent_groups': 3, 'independent_support_established': False,
        'status': 'posthoc_review_completed_independent_adjudication_unresolved_line_closed',
        'original_labels_or_statuses_changed': False, 'performance_recomputed': False,
        'new_forward_fit_download_sampling_or_contacts': 0,
    }
    write(OUT / 'FINAL_POSTHOC_REVIEW_AGGREGATES.json', summary)
    report = '''# Phase115最终提交评价

## 1．数据结果

用户导出的7条意见已原字节封存，按用户指示接收为最终，原状态2 reviewed／5 draft不修改。7条均自报落地云影、20m同质、上下文足够，7条有文字依据；只有2条记录不确定性，另5条为空。固定7例覆盖5个相关组：33VVE两例、44SKJ两例、40UEE／54TWN／19FDV各一例。字段完整的2例来自33VVE与54TWN两个组，字段完整不等于独立资格或语义正确已核验。

审阅编号、经验、接触史、身份核验均为unknown；导出时间按文件原值保存。根据本线程已发生的Phase114揭示和同一用户复核，本次是事后复核。新的合格独立判读0条／0组，未满足至少3个相关组的独立支持规则。原28图、Phase114意见SHA、原模板和v1/v2材料锁均核验未变。没有新前向、拟合、下载、抽样、标签修改或性能重算。

## 2．结论分析

本次完成了最终提交接收和事后复核评价。你的意见在固定五组中均继续支持候选为落地云影，但不能将7/7换算为总体正确率或漏检率，也不能称独立确认。5条不确定性缺项照实保留，不补填为“无”；2条已填的20m颜色不一致疑虑保留，不由程序替你否定或消除。

高光／暗区／厚云共现是提供的判读理由，不是已经核验太阳方向、云高投影或排除其他阴影的几何结论。本轮不据此认定低对比、排序损伤、域偏移或光谱不足机制。

按原协议，无合格独立资源时保持语义未裁决并收口该残余修复线索；本次不再要求原主审返工、不补样救场、不自动训练或改阈值。Phase112／113关闭结论保持。历史awaiting_independent_review材料状态保留为快照，当前状态以FINAL_POSTHOC_REVIEW_AGGREGATES.json为准。
'''
    with (OUT / 'FINAL_REVIEW_REPORT.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write(report)
    for name in ('FINAL_POSTHOC_REVIEW_AGGREGATES.json', 'FINAL_REVIEW_REPORT.md'):
        with (LOCAL / name).open('xb') as f:
            f.write((OUT / name).read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
