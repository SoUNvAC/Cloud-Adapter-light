"""Exclusive two-stage Phase87 delivery; lock image judgments before reference access."""
import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase87/phase87_20261011'
BASE = ROOT / 'outputs/phase65/alcd_visual_review_v1'
CODES = {'C': 'Cloud', 'H': 'Shadow', 'S': 'Surface', 'M': '混合', 'U': '无法判定'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def csvwrite(p, rows):
    assert rows
    with p.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs():
    frozen = read(OUT / 'SAMPLING_FROZEN.json')
    assert sha(OUT / 'sample_alias_lock.json') == frozen['sample_alias_lock_sha256']
    assert sha(OUT / 'blind_input_lock.json') == frozen['blind_input_lock_sha256']
    lock = read(OUT / 'sample_alias_lock.json')
    inputs = read(OUT / 'blind_input_lock.json')
    assert sha(ROOT / 'outputs/phase86/phase86_20261011/candidate_lock.json') == lock['candidate_lock_sha256']
    assert sha(BASE / 'sample_manifest.json') == lock['old_sample_manifest_sha256']
    for p in inputs['panels'] + inputs['sheets']:
        assert sha(OUT / p['path']) == p['sha256']
    for s in lock['samples']:
        assert sha(BASE / 'windows' / (s['source_id'] + '.npz')) == s['window_sha256']
        assert sha(BASE / 'images' / (s['source_id'] + '.png')) == s['original_blind_panel_sha256']
    return lock, inputs


def lock_blind():
    lock, inputs = validate_inputs()
    rows, viewed, batches = [], [], {}
    for p in sorted(OUT.glob('blind_batch_*.json')):
        batch = read(p)
        assert batch['stage'] == 'A'
        batches[p.name] = sha(p)
        viewed += batch['viewed_sheets']
        for r in batch['rows']:
            assert r['code'] in CODES and r['reason']
            rows.append(dict(anonymous_id=r['id'], judgment=CODES[r['code']],
                             reason=r['reason'], evidence_needed=r['needs'],
                             reviewer=batch['reviewer'], review_date='2026-10-11',
                             viewed_material='anonymous RGB + B08/B11/B12 + joint-valid mask; fixed original stretch/window',
                             panel_sha256=next(p['sha256'] for p in inputs['panels'] if p['id'] == r['id'])))
    ids = [r['anonymous_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 240
    assert set(ids) == {s['anonymous_id'] for s in lock['samples']}
    assert len(viewed) == len(set(viewed)) == 20
    assert set(viewed) == {Path(s['path']).name for s in inputs['sheets']}
    rows.sort(key=lambda r: r['anonymous_id'])
    csvwrite(OUT / 'blind_initial.csv', rows)
    result = dict(stage='A_complete_before_B', locked_utc=datetime.now(timezone.utc).isoformat(),
                  blind_initial_sha256=sha(OUT / 'blind_initial.csv'), batch_sha256=batches,
                  sample_lock_sha256=sha(OUT / 'sample_alias_lock.json'),
                  blind_input_lock_sha256=sha(OUT / 'blind_input_lock.json'),
                  reviewed_positions=len(rows), viewed_sheets=viewed,
                  judgments=dict(Counter(r['judgment'] for r in rows)))
    write(OUT / 'BLIND_INITIAL_LOCK.json', result)
    print(json.dumps(result, ensure_ascii=False))


def compare(judgment, reference):
    if not reference:
        return '参考缺失'
    if judgment in ('混合', '无法判定') or len(reference) != 1:
        return '仍无法裁决'
    return '一致' if judgment == reference[0] else '可能冲突'


def reveal():
    # No reference fields are read until the complete immutable A file is verified.
    sealed = read(OUT / 'BLIND_INITIAL_LOCK.json')
    assert sealed['stage'] == 'A_complete_before_B'
    assert sha(OUT / 'blind_initial.csv') == sealed['blind_initial_sha256']
    for name, digest in sealed['batch_sha256'].items():
        assert sha(OUT / name) == digest
    lock, inputs = validate_inputs()
    assert sha(OUT / 'sample_alias_lock.json') == sealed['sample_lock_sha256']
    assert sha(OUT / 'blind_input_lock.json') == sealed['blind_input_lock_sha256']
    with (OUT / 'blind_initial.csv').open(encoding='utf-8-sig', newline='') as f:
        initial = {r['anonymous_id']: r for r in csv.DictReader(f)}
    prior = {s['id']: s for s in read(BASE / 'sample_manifest.json')['samples']}
    expected = {k.replace(chr(92), '/'): v for k, v in read(BASE / 'SHA256.json').items()}
    old = {2: 'Cloud', 3: 'Cloud', 4: 'Shadow', 5: 'Surface', 6: 'Surface', 7: 'Surface'}
    revised = {64: 'Shadow', 128: 'Surface', 255: 'Cloud'}
    rows = []
    for s in lock['samples']:
        p = prior[s['source_id']]
        a = initial[s['anonymous_id']]
        original = [old[p['original_class']]]
        rc = p['revised_center']
        new = sorted({revised.get(v, 'invalid_' + str(v)) for v in rc}) if rc is not None else []
        panel = BASE / 'images' / (s['source_id'] + '_label.png')
        assert sha(panel) == expected[panel.relative_to(BASE).as_posix()]
        # Independently reconstruct the deterministic directed selections.
        for direction in (t for t in s['sources'] if t != 'uniform'):
            options = [r for r in prior.values() if r['product'] == s['alias'] and direction in r['groups']]
            winner = min(options, key=lambda r: hashlib.sha256(f"PHASE87|{s['product']}|{r['id']}".encode()).digest())
            assert winner['id'] == s['source_id']
            assert direction == original[0] + '_to_' + new[0] and len(new) == 1
        oc, nc = compare(a['judgment'], original), compare(a['judgment'], new)
        priority = 'P1' if ('Shadow' in original + new and (a['judgment'] in ('混合', '无法判定') or '可能冲突' in (oc, nc))) else 'P2' if '可能冲突' in (oc, nc) or a['judgment'] in ('混合', '无法判定') else 'P3'
        rows.append(dict(anonymous_id=s['anonymous_id'], group_id=s['group_id'], product=s['product'],
                         source_id=s['source_id'], sources=';'.join(s['sources']), row=s['row'], col=s['col'],
                         initial_judgment=a['judgment'], initial_reason=a['reason'], evidence_needed=a['evidence_needed'],
                         original_reference=original[0], revised_reference=';'.join(new) if new else '缺失',
                         original_comparison=oc, revised_comparison=nc, priority=priority,
                         disposition='待合格人工裁决；不改标签、不确认就绪',
                         reference_panel=str(panel.relative_to(ROOT)).replace(chr(92), '/'),
                         reference_panel_sha256=sha(panel), blind_panel_sha256=a['panel_sha256']))
    csvwrite(OUT / 'revealed_review.csv', rows)
    csvwrite(OUT / 'pending_adjudication.csv', sorted(rows, key=lambda r: (r['priority'], r['anonymous_id'])))
    summaries = []
    for alias in lock['aliases']:
        for stratum in ['uniform', 'difference']:
            subset = [r for r in rows if r['product'] == alias['representative'] and
                      (('uniform' in r['sources'].split(';')) if stratum == 'uniform' else any(t != 'uniform' for t in r['sources'].split(';')))]
            counts = Counter(r['initial_judgment'] for r in subset)
            summaries.append(dict(group_id=alias['group_id'], product=alias['representative'], stratum=stratum,
                                  positions=len(subset), decidable=sum(counts[c] for c in ['Cloud', 'Shadow', 'Surface']),
                                  mixed=counts['混合'], uncertain=counts['无法判定'], Cloud=counts['Cloud'],
                                  Shadow=counts['Shadow'], Surface=counts['Surface'],
                                  original_possible_conflict=sum(r['original_comparison'] == '可能冲突' for r in subset),
                                  revised_possible_conflict=sum(r['revised_comparison'] == '可能冲突' for r in subset),
                                  revised_missing=sum(r['revised_comparison'] == '参考缺失' for r in subset),
                                  larger_context=sum('larger_context' in r['evidence_needed'] for r in subset),
                                  clear_date=sum('clear_date' in r['evidence_needed'] for r in subset),
                                  DEM=sum('DEM' in r['evidence_needed'] for r in subset), confirmation_ready=0))
    csvwrite(OUT / 'group_summary.csv', summaries)
    write(OUT / 'group_summary.json', summaries)
    totals = {t: {k: sum(r[k] for r in summaries if r['stratum'] == t) for k in
                 ['positions', 'decidable', 'mixed', 'uncertain', 'Cloud', 'Shadow', 'Surface',
                  'original_possible_conflict', 'revised_possible_conflict', 'revised_missing']} for t in ['uniform', 'difference']}
    write(OUT / 'triage_summary.json', dict(groups=9, unique_positions=len(rows), strata=totals,
          missing_directions=len(lock['missing_directions']),
          empty_uniform_blocks=sum(a['missing_uniform_blocks'] for a in lock['aliases']),
          priorities=dict(Counter(r['priority'] for r in rows)), confirmation_ready_groups=0,
          stage_B_method='Rule comparison of locked AI center judgment with existing reference center categories; not truth adjudication',
          actual_stage_A_materials='20 anonymous contact sheets containing 240 RGB/falsecolor/valid-mask cards',
          stage_B_visual_QA='Recorded separately in VISUAL_QA.json after delivery preview'))
    build_html(rows)
    write(OUT / 'REVEAL_LOCK.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
          blind_initial_sha256=sha(OUT / 'blind_initial.csv'), revealed_review_sha256=sha(OUT / 'revealed_review.csv'),
          group_summary_sha256=sha(OUT / 'group_summary.csv'), stage_A_unchanged=True))
    print(json.dumps(totals, ensure_ascii=False))


def build_html(rows):
    content = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Phase87 标签风险初筛</title>',
      '<style>body{font-family:system-ui;margin:24px;background:#f4f5f6}article{background:white;padding:18px;margin:20px 0;border:1px solid #bbb}img{max-width:100%;height:auto}p{max-width:1200px}button{padding:8px}details{margin-top:12px}.P1{border-left:7px solid #b74726}</style>',
      '<h1>Phase87：AI辅助初筛，待人工裁决</h1><p>9组、240位置；确认就绪0。先完成匿名影像判断并锁定SHA，再进行参考中心类别对照。图中红框是60米中心格，不一定居中。灰色为无效观测。参考图左右分别为原版／已有修订版，白=Cloud、蓝=Shadow、绿=Surface；缺失/无效不是Surface。</p>',
      '<p>下方优先列出涉及云影且无法裁决或可能冲突的位置。所有240点完整保留。可能冲突不等于标签错误，AI与两版标签均不是真值；本轮不恢复人工复核、不运行模型。</p>',
      '<button onclick="document.querySelectorAll(\'details\').forEach(e=>e.open=true)">展开参考</button> <button onclick="document.querySelectorAll(\'details\').forEach(e=>e.open=false)">隐藏参考</button>']
    for r in sorted(rows, key=lambda r: (r['priority'], r['anonymous_id'])):
        e = {k: html.escape(str(v)) for k, v in r.items()}
        ref = '../../..' + '/' + r['reference_panel']
        content += [f'<article class="{e["priority"]}" id="{e["anonymous_id"]}"><h2>{e["anonymous_id"]} · {e["priority"]} · {e["group_id"]}</h2>',
          f'<p>{e["product"]}<br>来源：{e["sources"]}；原位置：{e["source_id"]}</p>',
          f'<img loading="lazy" src="blind_panels/{e["anonymous_id"]}.png" alt="匿名影像">',
          f'<p><b>锁定的影像判断：{e["initial_judgment"]}</b>。{e["initial_reason"]}<br>可能需要：{e["evidence_needed"]}</p>',
          f'<details><summary>揭示参考（独立于已锁定判断）</summary><p>原版：{e["original_reference"]} → {e["original_comparison"]}；修订版：{e["revised_reference"]} → {e["revised_comparison"]}</p><img loading="lazy" src="{html.escape(ref)}" alt="原版和修订版参考"><p>{e["disposition"]}</p></details></article>']
    content.append('</html>')
    with (OUT / 'index.html').open('x', encoding='utf-8') as f:
        f.write('\n'.join(content))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['lock-blind', 'reveal'])
    args = parser.parse_args()
    lock_blind() if args.stage == 'lock-blind' else reveal()
