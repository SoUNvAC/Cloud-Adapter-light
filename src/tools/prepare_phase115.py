# Fixed seven-case material preparation; no independent adjudication.
import json, csv, hashlib, collections, datetime, shutil, zipfile, re, html
from pathlib import Path
from PIL import Image
R = Path(__file__).resolve().parents[2]
O = R / 'src/research_results/phase115_20261014'
U = R / 'outputs/phase115/phase115_20261014'
D = R / 'data/pixbox_s2_cmix_5036991/phase115_20261014'
P = R / 'src/research_results/phase114_20261014'
OLD = R / 'outputs/phase114/phase114_20261014'
DATA = R / 'data/pixbox_s2_cmix_5036991/phase114_20261014'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, x):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(x, f, indent=2, ensure_ascii=False)
        f.write('\n')

def main():
    assert not O.exists() and (not U.exists()) and (not D.exists()), 'Preserve prior attempt'
    meta = read(P / 'LOCAL_ARTIFACT_SHA.json')
    original = DATA / 'SEALED_BLIND_OPINIONS.json'
    assert sha(original) == meta['original_opinion_file_sha256'] == '552f12c66a15f34a306104a70b2ad9bea7a7717d2261f9e9292c1f58a699b9a6'
    statuses = collections.Counter((x['status'] for x in read(original)['records']))
    assert statuses == {'reviewed': 1, 'draft': 46}
    source = DATA / 'REVEALED_FINAL_CANDIDATES_PRIVATE.json'
    assert sha(source) == meta['private_final_reveal_sha256']
    seven = [x for x in read(source)['rows'] if x['reference_and_fixed_decision']['sampling_layer'] == 'FN' and x['opinion']['point_candidate_class'] == 'Shadow']
    assert len(seven) == 7 and {x['opinion']['anonymous_id'] for x in seven} == {'A002', 'A019', 'A023', 'A029', 'A037', 'A046', 'A047'}
    groups = collections.Counter((x['reference_and_fixed_decision']['product_id'].split('_T')[1][:5] for x in seven))
    assert groups == {'33VVE': 2, '44SKJ': 2, '40UEE': 1, '54TWN': 1, '19FDV': 1}
    with (R / 'data/pixbox_s2_cmix_5036991/phase103_20261013/coordinate_metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        coords = {x['point_id']: x for x in csv.DictReader(f)}
    grids = {x['product_id']: x['grid20'] for x in read(R / 'src/research_results/phase104_20261013/object_manifest.json')['products']}
    oldgeo = read(OLD / 'RENDER_ACCEPTANCE.json')['geometry']
    seven.sort(key=lambda x: hashlib.sha256(('PHASE115_INDEPENDENT_ORDER|' + x['reference_and_fixed_decision']['point_id']).encode()).hexdigest())
    trace = []
    files = {}
    geometry = []
    for i, x in enumerate(seven, 1):
        aid = f'J{i:03d}'
        oldid = x['opinion']['anonymous_id']
        ref = x['reference_and_fixed_decision']
        c = coords[ref['point_id']]
        g = grids[ref['product_id']]
        for k in ['product_id', 'group_id', 'row20', 'col20']:
            assert str(ref[k]) == str(c[k])
        fx = (float(c['native_x_m']) - g['ULX']) / 20
        fy = (g['ULY'] - float(c['native_y_m'])) / 20
        assert int(fx) == int(c['col20']) and int(fy) == int(c['row20']) and (int(fx * 2) == int(c['col10'])) and (int(fy * 2) == int(c['row10']))
        trace.append({'independent_id': aid, 'old_anonymous_id': oldid, 'point_id': ref['point_id'], 'product_id': ref['product_id'], 'group_id': ref['group_id'], 'row10': int(c['row10']), 'col10': int(c['col10']), 'row20': int(c['row20']), 'col20': int(c['col20']), 'native_x_m': float(c['native_x_m']), 'native_y_m': float(c['native_y_m']), 'old_opinion_sha256': sha(original), 'old_reveal_sha256': sha(source)})
        for width in [640, 10240]:
            q = next((z for z in oldgeo if z['anonymous_id'] == oldid and z['extent_m'] == width))
            assert q['true_point_inside_original_20m_cell'] and q['same_geographic_bounds'] and q['native_RGB_point_not_replaced_by_cell_center']
            for kind in ['RGB', 'FALSE']:
                oldname = f'{oldid}_{width}_{kind}.png'
                p = OLD / 'assets' / oldname
                digest = sha(p)
                assert digest == meta['anonymous_image_sha256'][oldname]
                files[f'{aid}_{width}_{kind}.png'] = {'source_path': p, 'sha256': digest}
                with Image.open(p) as im:
                    assert im.size == (1024, 1088) and (not any((oldid in str(v) for v in im.info.values())))
            geometry.append({'independent_id': aid, 'extent_m': width, 'original_10m_point_equal': True, 'original_20m_cell_equal': True, 'original_geographic_bounds_equal': True})
    O.mkdir(parents=True)
    U.mkdir(parents=True)
    D.mkdir(parents=True)
    (O / '.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
    write(D / 'TRACE_PRIVATE.json', {'rows': trace, 'do_not_supply_to_independent_reviewer': True})
    sources = [Path(__file__), R / 'src/research_plans/PHASE115_SEVEN_CASE_ADJUDICATION_PACK_20261014.md', P / 'FINAL_REVIEW_AGGREGATES.json', P / 'LOCAL_ARTIFACT_SHA.json', P / 'SUBMISSION_ACCEPTANCE.json', P / 'NEXT_HYPOTHESIS.md']
    write(O / 'SEVEN_CASE_ANON_LOCK.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': {p.relative_to(R).as_posix(): sha(p) for p in sources}, 'cases': 7, 'groups': dict(groups), 'images': 28, 'trace_sha256': sha(D / 'TRACE_PRIVATE.json'), 'original_opinion_sha256': sha(original), 'original_reveal_sha256': sha(source), 'image_sha256': {n: x['sha256'] for n, x in files.items()}, 'order_prefix': 'PHASE115_INDEPENDENT_ORDER|', 'same_fixed_seven_no_substitution': True, 'independent_review_status': 'awaiting_independent_review', 'new_sampling': 0, 'minimum_support_rule': 'at least3 distinct related groups each with>=1 independent ground-cloud-shadow, homogeneous20m, complete evidence, decidable'})
    pack = U / 'reviewer_pack'
    (pack / 'assets').mkdir(parents=True)
    for n, x in files.items():
        shutil.copyfile(x['source_path'], pack / 'assets' / n)
        assert sha(pack / 'assets' / n) == x['sha256']
    template = {'reviewer_id': 'unknown', 'reviewer_experience': 'unknown', 'has_seen_original_reveal': 'unknown', 'identity_verification': 'unknown', 'actual_submitted_at': None, 'allowed_point_classes': ['Cloud', '落地云影', '其他阴影或暗地表', '混合', '无法判断'], 'allowed_cell_states': ['同质', '混合', '无法判断'], 'records': [{'independent_id': x['independent_id'], 'status': 'not_reviewed', 'point_candidate_class': None, 'cell_20m_state': None, 'visible_evidence': None, 'uncertainty': None, 'existing_context_sufficient': 'unknown'} for x in trace]}
    write(pack / 'independent_adjudication_template.json', template)
    instructions = '# 独立影像判读\n\n按可见证据逐点填写原10m位置类别、20m单元同质性、具体依据、不确定性及现有上下文是否足够。无法判断如实填写，不预设类别。黄色为原点，青色为20m单元；灰色无观测或影像外，粉色饱和。固定TOA0—0.3与最近邻显示，放大不增加细节。\n\n打开index.html看图，填写independent_adjudication_template.json；另存为independent_adjudication_submission.json。记录审阅者编号、经验、是否接触原参考或先前揭示，以及实际提交时间（ISO时间字符串）。未知保留unknown，不能伪称专家身份已经核验。\n\n如果已接触原揭示，后续只能称事后复核。先不查原参考或其他人的意见；意见SHA封存前不揭示对照。没有实际判读不填写结论。\n'
    (pack / 'README_FOR_REVIEWER.md').write_text(instructions, encoding='utf-8', newline='\n')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立影像判读材料</title><style>body{font-family:system-ui;margin:25px;background:#f6f7fa;color:#182333}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}figure{background:white;padding:8px;margin:0}img{width:100%;cursor:zoom-in}nav{position:sticky;top:0;background:white;padding:12px}nav a{margin:12px}section{margin-top:25px}</style><h1>独立影像判读材料</h1><p>黄色十字为原10m点，青色框为20m单元；灰色无观测／影像外，粉色饱和。固定TOA0—0.3；最近邻放大不增加细节。点击图片看完整尺寸。</p><p>按可见依据填写，不确定如实记录。<a href="independent_adjudication_template.json" download>下载空白裁决模板</a> · <a href="README_FOR_REVIEWER.md">填写说明</a></p><nav>'
    page += ''.join((f"""<a href="#{x['independent_id']}">{x['independent_id']}</a>""" for x in trace)) + '</nav>'
    for x in trace:
        aid = x['independent_id']
        page += f'<section id="{aid}"><h2>{aid}</h2><div class="grid">'
        for width, kind, label in [(640, 'RGB', '640m：原生10m RGB'), (640, 'FALSE', '640m：原生20m NIR/R/G'), (10240, 'RGB', '10.24km：原生10m RGB'), (10240, 'FALSE', '10.24km：原生20m NIR/R/G')]:
            page += f'<figure><figcaption>{label}</figcaption><a href="assets/{aid}_{width}_{kind}.png" target="_blank"><img alt="{aid} {label}" src="assets/{aid}_{width}_{kind}.png"></a></figure>'
        page += '</div></section>'
    page += '</html>'
    assert not re.search('\\b(FN|FP|Omni|Source|group_id|product_id|sampling_layer|shadow_confidence|A[0-9]{3})\\b', page)
    (pack / 'index.html').write_text(page, encoding='utf-8', newline='\n')
    with zipfile.ZipFile(U / 'independent_review_pack.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pack.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(pack).as_posix())
    write(O / 'PACKAGE_SHA.json', {'archive_sha256': sha(U / 'independent_review_pack.zip'), 'archive_bytes': (U / 'independent_review_pack.zip').stat().st_size, 'reviewer_artifacts_sha256': {p.relative_to(pack).as_posix(): sha(p) for p in pack.rglob('*') if p.is_file()}, 'private_trace_sha256': sha(D / 'TRACE_PRIVATE.json')})
    write(O / 'CPU_MATERIAL_ACCEPTANCE.json', {'status': 'passed', 'cases': 7, 'groups': 5, 'images': 28, 'exact_same_seven_IDs': True, 'geometry': geometry, 'all28_original_image_SHA_equal': True, 'anonymous_filenames_and_page_passed': True, 'PNG_metadata_has_no_original_alias': True, 'no_old_opinion_reference_score_group_or_selection_reason_in_reviewer_material': True, 'all7_independent_opinions_blank': True, 'independent_human_reviews': 0, 'new_forward_fit_download_or_sampling': 0})
    write(O / 'MISSING_FIELDS_PRESERVED.json', {'original_opinion_sha256': sha(original), 'original_raw_statuses': dict(statuses), 'missing_evidence': 24, 'missing_context': 46, 'missing_human_arbitration': 46, 'original_final_opinion_not_modified': True, 'original_reviewer_not_asked_to_refill': True, 'independent_reviewer_id': 'unknown', 'independent_reviewer_experience': 'unknown', 'independent_exposure': 'unknown', 'independent_actual_submission_time': None, 'independent_identity_verification': 'unknown'})
    print('Fixed7 cases,28 identical images,0 independent opinions; awaiting_independent_review')
if __name__ == '__main__':
    main()
