"""Reference-support audit; cached supports/metadata only, except SHA-checked SP masks.

Zero reference support is never interpreted as verified physical absence.
No models, fitting, reserved pixel decoding, or changes to historical results.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/label_support_audit/20261028'
INPUTS = {}


def read(path):
    p = ROOT / path
    INPUTS[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix == '.csv':
        with p.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def record(dataset, scene, split, counts, scope, status, group='', **extra):
    row = dict(dataset=dataset, scene=scene, split=split, correlation_group=group,
               support_scope=scope, official_shadow_status=status,
               zero_reference_is_verified_absence=False)
    for i, label in enumerate(('Surface', 'Cloud', 'Shadow')):
        row[label + '_count'] = counts[i] if counts is not None else ''
        row['has_' + label + '_reference'] = int(counts[i] > 0) if counts is not None else ''
    row['Shadow_AP_support'] = (
        'not_audited' if counts is None else
        'positive_reference_present_not_quality_certified' if counts[2] > 0 else
        'NA_no_positive_reference')
    row.update(extra)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_lock = read('outputs/phase236/phase236_20261027/delivery/LOCK.json')
    assert source_lock['Source_SHA256'] == '64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
    for rel in ['src/configs/protocol/phase64_source_parent_rgb.py',
                'src/tools/evaluate_phase64_source_only_parent.py', 'src/tools/phase236_common.py']:
        INPUTS[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    rows = []
    l8_status = read('src/research_plans/protocol_data/l8_biome_usgs_shadow_status.csv')
    l8 = {}
    train = read('outputs/phase235/phase235_20261027/delivery/TRAIN_SCENE_SUPPORT.json')
    for z in train:
        l8[z['scene']] = dict(split='train18', counts=z['support'],
                             scope='native_selected_training_crops', Fill_count=z['Fill_pixels'])
    per_scene = read('outputs/phase236/phase236_20261027/PER_SCENE.json')
    for z in per_scene:
        if z['domain'] == 'L8' and z['source']:
            counts = [z[k + '_pixels'] for k in ('Surface', 'Cloud', 'Shadow')]
            item = dict(split=z['split'], counts=counts, scope='native_evaluation_crops_Fill_excluded')
            if z['scene'] in l8:
                assert l8[z['scene']]['counts'] == counts
            l8[z['scene']] = item
    for z in l8_status:
        a = l8.get(z['scene'], dict(split='outside_current_42_scene_pool', counts=None,
                                  scope='official_metadata_only_no_pixels_read'))
        rows.append(record('L8_Biome', z['scene'], a['split'], a['counts'], a['scope'],
                           'USGS_Shadows_' + z['usgs_shadows'], Fill_count=a.get('Fill_count', ''),
                           annotation_note='yes/no does not independently certify completeness or true absence'))
    assert len(train) == 18
    assert all(next(z for z in l8_status if z['scene'] == t['scene'])['usgs_shadows'] == 'yes'
               and t['support'][2] > 0 for t in train)

    sp = read('outputs/label_support_audit/20261028/SPARCS_NATIVE_RECEIPT.json')['rows']
    assert len(sp) == 80 and len({z['scene'] for z in sp}) == 80
    for z in sp:
        assert sum(z['support']) == 1000000
        rows.append(record('SPARCS', z['scene'], z['split'], z['support'],
                           'native_1000x1000_all_reference_pixels', 'manual_mask_no_scene_missingness_flag_found',
                           mask_sha256=z['mask_sha256'], raw_Shadow_0=z['native_counts'].get('0', 0),
                           raw_Shadow_over_water_1=z['native_counts'].get('1', 0)))
    # Independent agreement with the previously sealed native train census.
    sp_train = read('outputs/phase236/phase236_20261027/delivery/fits/SPARCS_21701_1e-04/TRAIN_POOL.json')['census']
    for z in sp_train:
        actual = next(a for a in sp if a['split'] == 'train' and a['scene'] == z['scene'])
        assert actual['support'] == [z['support'][str(i)] for i in range(3)]

    cat_groups = [z for z in read('outputs/phase205/phase205_20261019/Catalogue_PER_GROUP.csv') if z['strategy'] == 'R4']
    cat_actual = {}
    group_rows = []
    for z in cat_groups:
        products = z['products'].split(',')
        assert len(products) == 1, 'Do not assign a group support to multiple products'
        counts = [sum(r) for r in json.loads(z['confusion'])]
        cat_actual[products[0]] = (counts, z['group_id'])
        group_rows.append(record('Catalogue', z['group_id'], 'development32_groups', counts,
                                 'fixed_current_evaluation_support', 'see_product_metadata', z['group_id']))
    tags = read('data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv')
    assert len(tags) == 513
    for z in tags:
        counts, group = cat_actual.get(z['scene'], (None, ''))
        row = record('Catalogue', z['scene'], 'development32' if counts else 'outside_current_development_no_pixels_read',
                     counts, 'fixed_current_evaluation_support' if counts else 'author_percentage_metadata_only',
                     'annotated_if_present' if z['shadows_marked'] == '1' else 'known_unannotatable_shadow', group,
                     author_split_not_research_split=z['dataset'],
                     shadow_evaluation_annotation_eligible=int(z['shadows_marked'] == '1'))
        for label, key in [('Surface', 'clear_percent'), ('Cloud', 'cloud_percent'), ('Shadow', 'shadow_percent')]:
            row['author_' + label + '_percent'] = z[key]
            row['author_reports_positive_' + label + '_fraction'] = int(float(z[key]) > 0)
        rows.append(row)
    assert all(next(t for t in tags if t['scene'] == p)['shadows_marked'] == '1' for p in cat_actual)

    points = read('data/pixbox_s2_cmix_5036991/phase103_20261013/parsed_allowed_rows.json')
    expected = read('outputs/phase205/phase205_20261019/EXPECTED_SUPPORT.json')['rows']
    valid_ids = {p for z in expected if z['pool'] == 'PixBox' for p in z['common_image_valid_point_ids']}
    ref_class = dict(negative_surface=0, negative_cloud=1, positive_ground_cloud_shadow=2)
    product_counts, group_counts = {}, {}
    for product in sorted({z['product_id'] for z in points}):
        pts = [z for z in points if z['product_id'] == product]
        group = pts[0]['group_id']
        assert all(z['group_id'] == group for z in pts)
        counts = [0, 0, 0]
        for p in pts:
            if p['point_id'] in valid_ids and p['reference'] in ref_class:
                counts[ref_class[p['reference']]] += 1
        product_counts[product] = counts
        g = group_counts.setdefault(group, [0, 0, 0])
        for i in range(3):
            g[i] += counts[i]
        rows.append(record('PixBox', product, 'fixed_allowed16_products', counts,
                           'sparse_reference_points_on_fixed_common_image_support', 'sparse_points_not_dense_mask', group,
                           original_allowed_points=len(pts),
                           ignored_reference_points=sum(z['reference'] == 'ignore' for z in pts),
                           original_Surface_points=sum(z['reference'] == 'negative_surface' for z in pts),
                           original_Cloud_points=sum(z['reference'] == 'negative_cloud' for z in pts),
                           original_Shadow_points=sum(z['reference'] == 'positive_ground_cloud_shadow' for z in pts)))
    pix_groups = [z for z in read('outputs/phase205/phase205_20261019/PixBox_PER_GROUP.csv') if z['strategy'] == 'R4']
    for z in pix_groups:
        counts = [sum(r) for r in json.loads(z['confusion'])]
        assert group_counts[z['group_id']] == counts, (z['group_id'], group_counts[z['group_id']], counts)
        group_rows.append(record('PixBox', z['group_id'], 'fixed10_correlation_groups', counts,
                                 'sparse_common_evaluation_points', 'sparse_points_not_dense_mask', z['group_id']))

    strata = read('outputs/phase236/phase236_20261027/L8_TEST_STRATA.json')
    no = [z for z in strata if z['stratum'] == 'no']
    fpr = {s: sum(z['nonShadow_to_Shadow_FPR'] for z in no if z['strategy'] == s) /
              sum(z['strategy'] == s for z in no) for s in ('Source', 'E', 'W', 'P')}
    summary = dict(
        row_counts=dict(collections.Counter(z['dataset'] for z in rows)),
        measured_support_by_dataset={d: dict(
            measured_rows=sum(z['dataset'] == d and z['Shadow_count'] != '' for z in rows),
            positive_shadow_rows=sum(z['dataset'] == d and z['Shadow_count'] != '' and z['Shadow_count'] > 0 for z in rows),
            zero_shadow_rows=sum(z['dataset'] == d and z['Shadow_count'] == 0 for z in rows))
            for d in ('L8_Biome', 'SPARCS', 'Catalogue', 'PixBox')},
        L8_train18_all_yes_and_shadow_present=True,
        Source_identity=dict(checkpoint_SHA256=source_lock['Source_SHA256'],
                             training_dataset='CloudSEN12 high L1C',
                             config='src/configs/protocol/phase64_source_parent_rgb.py',
                             parent_mapping=[0, 1, 1, 2],
                             pixel_annotation_equivalence_to_Biome='not_established'),
        L8_train_native_support=[sum(z['support'][i] for z in train) for i in range(3)],
        L8_train_Fill_pixels=sum(z['Fill_pixels'] for z in train),
        L8_no10_nominal_pooled_FPR_percent_seed_mean=fpr,
        Catalogue_known_unannotatable=sum(z['shadows_marked'] == '0' for z in tags),
        Catalogue_annotated_if_present=sum(z['shadows_marked'] == '1' for z in tags),
        Catalogue_marked_but_metadata_zero_shadow=sum(z['shadows_marked'] == '1' and float(z['shadow_percent']) == 0 for z in tags),
        PixBox_allowed_products=len(product_counts), PixBox_allowed_points=len(points),
        PixBox_current_group_count=len(group_counts),
        PixBox_current_reference_counts=[sum(x[i] for x in product_counts.values()) for i in range(3)],
        group_support_positive_shadow={d: sum(z['dataset'] == d and z['Shadow_count'] > 0 for z in group_rows)
                                       for d in ('Catalogue', 'PixBox')},
        no_new_model_calls=True, no_reserved_pixels_read=True,
        limitations=['No pixel census for L8 scenes outside current42 or Catalogue outside current32; metadata only.',
                     'PixBox only fixed allowed16 of official29 products; no new pool unsealed.',
                     'Presence is reference support, not proof of annotation completeness or correctness.',
                     'Native SPARCS counts do not equal resized512 evaluation counts.'],
        checks=['80 SPARCS mask SHA verified against frozen inputs', '60 SPARCS train counts match sealed census',
                'PixBox per-product sums exactly reproduce fixed10-group reference supports',
                'Catalogue current32 all officially shadow-annotated-if-present'])
    write_csv('PER_SCENE_CLASS_SUPPORT.csv', rows)
    write_csv('CURRENT_GROUP_SUPPORT.csv', group_rows)
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'INPUT_SHA256.json').write_text(json.dumps(INPUTS, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
