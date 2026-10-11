"""Build preview QA sheets, then seal the local Phase87 delivery."""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from finalize_phase87 import ROOT, OUT, read, sha, write, csvwrite, validate_inputs


def preview():
    from PIL import Image, ImageDraw
    with (OUT / 'revealed_review.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    lock = read(OUT / 'sample_alias_lock.json')
    selected = []
    for alias in lock['aliases']:
        candidates = [r for r in rows if r['product'] == alias['representative']]
        selected.append(min(candidates, key=lambda r: (r['priority'], r['anonymous_id'])))
    sheets = []
    for start in range(0, 9, 3):
        sheet = Image.new('RGB', (880, 3 * 940), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, r in enumerate(selected[start:start+3]):
            y = n * 940
            sheet.paste(Image.open(OUT / 'blind_panels' / (r['anonymous_id'] + '.png')), (0, y))
            label = f"{r['anonymous_id']} | Original: {r['original_reference']} | Revised: {r['revised_reference']}"
            # ASCII fallback for an absent revised reference, no font dependency.
            label = label.encode('ascii', errors='replace').decode('ascii')
            draw.text((10, y + 475), label, fill='black')
            sheet.paste(Image.open(ROOT / r['reference_panel']), (0, y + 500))
        path = OUT / f'reveal_preview_{start//3+1}.png'
        assert not path.exists()
        sheet.save(path)
        sheets.append(dict(path=path.name, ids=[r['anonymous_id'] for r in selected[start:start+3]], sha256=sha(path)))
    write(OUT / 'preview_manifest.json', sheets)
    print(json.dumps(sheets))


def seal():
    lock, inputs = validate_inputs()
    a = read(OUT / 'BLIND_INITIAL_LOCK.json')
    b = read(OUT / 'REVEAL_LOCK.json')
    assert sha(OUT / 'blind_initial.csv') == a['blind_initial_sha256'] == b['blind_initial_sha256']
    assert sha(OUT / 'revealed_review.csv') == b['revealed_review_sha256']
    qa = read(OUT / 'VISUAL_QA.json')
    manifest = read(OUT / 'preview_manifest.json')
    assert qa['viewed_preview_ids'] == [i for s in manifest for i in s['ids']]
    assert len(qa['viewed_preview_ids']) == 9
    for s in manifest:
        assert sha(OUT / s['path']) == s['sha256']
    with (OUT / 'revealed_review.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len({r['anonymous_id'] for r in rows}) == 240
    assert len({r['product'] for r in rows}) == 9
    # Validate every HTML image target, including references kept at their original location.
    page = (OUT / 'index.html').read_text(encoding='utf-8')
    targets = re.findall(r'<img[^>]+src="([^"]+)"', page)
    assert len(targets) == 480 and all((OUT / t).resolve().is_file() for t in targets)
    # Exact uniform membership and direction cap, independently checked against old sampling.
    prior = read(ROOT / 'outputs/phase65/alcd_visual_review_v1/sample_manifest.json')['samples']
    for alias in lock['aliases']:
        expected = {r['id'] for r in prior if r['product'] == alias['alias'] and 'uniform' in r['groups']}
        selected = [r for r in rows if r['product'] == alias['representative']]
        assert {r['source_id'] for r in selected if 'uniform' in r['sources'].split(';')} == expected
        direction_counts = Counter(t for r in selected for t in r['sources'].split(';') if t != 'uniform')
        assert all(n == 1 for n in direction_counts.values())
    summary = read(OUT / 'triage_summary.json')
    assert summary['confirmation_ready_groups'] == 0
    csvwrite(OUT / 'stage_B_evidence_needs.csv', [dict(anonymous_id=r['anonymous_id'],
        initial_evidence_needed=r['evidence_needed'],
        after_reveal_evidence_needed=(r['evidence_needed'] if r['evidence_needed'] != 'none' else
            'larger_context;clear_date' if '可能冲突' in (r['original_comparison'], r['revised_comparison']) else
            'qualified_human_adjudication'),
        visual_QA_note=qa['observations'].get(r['anonymous_id'], 'Not separately visually adjudicated in Stage B; locked-judgment/reference-center rule comparison'),
        pending='yes') for r in rows])
    text = '''# Phase87交付：ALCD九代表标签风险初筛

## 1．数据结果

固定Phase86九组与九代表，复用240个既有窗口：210常规点、30差异点。15个空常规区块未补抽；54个可能的有向差异层中24层缺失，未补抽。两种来源本次没有重复点。

| 来源 | 位置数 | 可给出单类判断 | 混合 | 无法判定 | 与原版可能冲突 | 与修订版可能冲突 | 修订参考缺失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 常规 | 210 | 86 | 47 | 77 | 7 | 6 | 70 |
| 差异 | 30 | 13 | 4 | 13 | 8 | 8 | 0 |

常规单类判断：Cloud 9、Surface 77、Shadow 0；差异单类判断：Cloud 2、Surface 11、Shadow 0。这里的“单类判断”只是AI影像解释，不代表参考标签已验证。修订参考缺失集中于三个代表的常规点。

Stage A实际查看20张匿名联系表，覆盖全部240位置的RGB、B08/B11/B12假彩色及有效mask。先锁完整CSV及SHA，再读取参考。判读者：AI GPT-6 / Codex；精确运行快照未暴露，未伪造版本。日期2026-10-11。

Stage B对全部240点做锁定判断与参考中心类别的规则对照，页面提供原版／修订版参考图。另对九代表各一张优先证据卡做图像排版QA，实际查看ID在VISUAL_QA.json。未声称逐点重新作空间语义裁决。优先级P1为30点、P2为132点、P3为78点；全部240点仍保留待合格人工裁决状态。

## 2．结论探讨

现有小窗口不足以稳妥区分暗地表、地形阴影和云影；60米格跨边界时保留混合。Shadow单类判断为0，表示本轮未获得足够影像证据支持这种判断，不表示数据没有云影，也不能据此证明参考云影错误。较低可判定性也限制了这次初筛的能力。

可能冲突来自锁定AI解释与参考类别不一致，不是标签错误率。差异点经过定向抽样，不能与常规点混合估计总体准确率；原版、修订版和AI都没有自动真值地位。不改标签、不删组、不换代表、不选择参考版本。

更大上下文、晴空日期和DEM是后续可能需要的证据；清单中的DEM标记是潜在需求，不代表每一点必须使用DEM。本轮不追加影像、不训练、不运行分割模型、不读取旧确认池像元或预测。人工复核继续暂停，确认就绪组数仍为0。九代表均有参考云影，该池不能单独支撑一般场景组合的1%平均误报预算。

## 文件使用

- [index.html](index.html)：240张证据卡，P1优先；默认隐藏参考，点击展开。影像复用既有原文件，不重复打包。
- [blind_initial.csv](blind_initial.csv)：不可覆盖的Stage A判断；[BLIND_INITIAL_LOCK.json](BLIND_INITIAL_LOCK.json)保存锁定SHA。
- [revealed_review.csv](revealed_review.csv)：参考中心类别与判断对照；[pending_adjudication.csv](pending_adjudication.csv)为完整待裁决清单。
- [group_summary.csv](group_summary.csv)：9代表×常规／差异18行，含逐组证据需求。
- stage_B_evidence_needs.csv：揭示后补充的证据需求，保持A阶段需求和判断不变；VISUAL_QA.json记录实际查看的九张参考卡。
- sample_alias_lock.json、blind_input_lock.json、SAMPLING_FROZEN.json：身份、抽样、原窗口／输入哈希和锁定时间。
- DELIVERY_SHA256.json：本次本地交付文件哈希；LOCAL_VERIFIED.json：成员、盲审锁、抽样、图像链接及QA校验。

页面引用同一仓库内outputs/phase65/alcd_visual_review_v1/images的既有标签图；移动交付目录时需保留此相对路径。旧人工模板和原参考图未修改；无需交回新的人工CSV。
'''
    with (OUT / 'README.md').open('x', encoding='utf-8') as f:
        f.write(text)
    write(OUT / 'LOCAL_VERIFIED.json', dict(verified_utc=datetime.now(timezone.utc).isoformat(),
          checks=['nine_fixed_representatives', '240_unique_positions', 'uniform_exact_membership',
                  'directed_selection_SHA256_winners_and_caps', 'Stage_A_complete_before_reference_reveal',
                  'Stage_A_CSV_and_batch_hashes_unchanged', 'all_original_window_and_blind_input_hashes',
                  '480_HTML_image_targets_exist', 'nine_representative_preview_QA'],
          stage_A_sha256=a['blind_initial_sha256'], confirmation_ready_groups=0))
    for name, digest in a['batch_sha256'].items():
        assert sha(OUT / name) == digest
    write(OUT / 'DELIVERY_SHA256.json', {str(p.relative_to(OUT)).replace(chr(92), '/'): sha(p)
          for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps(dict(delivery_files=len(read(OUT / 'DELIVERY_SHA256.json')), unique_positions=240,
          ready=0, stage_A_sha256=sha(OUT / 'blind_initial.csv')), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preview', 'seal'])
    args = p.parse_args()
    preview() if args.action == 'preview' else seal()
