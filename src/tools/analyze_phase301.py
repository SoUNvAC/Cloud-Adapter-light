"""Read sealed Phase300 caches; explain prediction transitions without inference."""
import csv
import gc
import io
import json
import os
from pathlib import Path
import time
import sys
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import phase300_reference as ref
import numpy as np

R = Path(__file__).resolve().parents[2]
REMOTE = os.environ.get('PHASE301_REMOTE') == '1'
ROOT = Path('/home/scv/shared/phase301_20261020') if REMOTE else R/'outputs/phase301/phase301_20261020'
P = R/'src/research_results/phase300_20261020' if REMOTE else R/'outputs/phase300/phase300_20261020'
D = Path('/home/scv/shared/phase300_20261020') if REMOTE else P/'remote_backup/phase300_20261020'
O = ROOT/'verified_run'
DATA = ROOT/'inputs' if REMOTE else R
FIT_SOURCE = Path('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json') if REMOTE else R/'outputs/phase208/phase208_20261020/12201_FIT_INPUT_LOCK.json'
if REMOTE:
    ref.R = DATA
    ref.OLD = Path('/home/scv/shared/phase205_20261019')
    ref.D = D
S = R/'src/research_results/phase301_20261020/verified_run'
CLASSES = ('Surface', 'Cloud', 'Shadow')
INPUTS = {}
BASE_SHA = ref.sha
CHECKS = 0
POOLED = {}
PRODUCT_CELLS = []
STRATA_CHECKS = 0


def sha(path):
    value = BASE_SHA(path)
    INPUTS[str(Path(path).resolve())] = value
    return value


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(name, rows):
    assert rows, name
    with (O/name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'NA' if v is None else v for k, v in row.items()})


def tensor(y, a, b):
    global CHECKS
    t = np.bincount((9*y+3*a+b).astype(np.int64), minlength=27).reshape(3, 3, 3)
    assert int(t.sum()) == len(y)
    for cls in range(3):
        for old in range(3):
            for new in range(3):
                direct = np.count_nonzero((y == cls) & (a == old) & (b == new))
                assert int(t[cls, old, new]) == int(direct)
                CHECKS += 1
    assert np.array_equal(t.sum(2), np.bincount((3*y+a).astype(np.int64), minlength=9).reshape(3, 3))
    assert np.array_equal(t.sum(1), np.bincount((3*y+b).astype(np.int64), minlength=9).reshape(3, 3))
    return t


def cells(t, prefix):
    rows = []
    for cls in range(3):
        support = int(t[cls].sum())
        for old in range(3):
            for new in range(3):
                n = int(t[cls, old, new])
                rows.append(dict(**prefix, truth=CLASSES[cls], old_prediction=CLASSES[old],
                                 new_prediction=CLASSES[new], class_support=support, count=n,
                                 class_fraction_pct=100*n/support if support else None))
    return rows


def errors(t, prefix):
    rows = []
    for cls in range(3):
        support = int(t[cls].sum())
        for wrong in range(3):
            if wrong == cls:
                continue
            old = int(t[cls, wrong].sum())
            new = int(t[cls, :, wrong].sum())
            kept = int(t[cls, wrong, wrong])
            removed, added = old-kept, new-kept
            corrected = int(t[cls, wrong, cls])
            assert removed-added == old-new
            assert 0 <= corrected <= removed
            rows.append(dict(**prefix, error=CLASSES[cls]+'->'+CLASSES[wrong], class_support=support,
                             R4_errors=old, student_errors=new, removed=removed, added=added,
                             net_reduction=old-new, corrected_among_removed=corrected,
                             changed_to_other_error=removed-corrected,
                             error_rate_change_pp=100*(new-old)/support if support else None))
    return rows


def states(t, prefix):
    rows = []
    for cls in range(3):
        support = int(t[cls].sum())
        counts = dict(correct_retained=int(t[cls, cls, cls]),
            new_error=sum(int(t[cls, cls, j]) for j in range(3) if j != cls),
            corrected=sum(int(t[cls, j, cls]) for j in range(3) if j != cls),
            same_old_error=sum(int(t[cls, j, j]) for j in range(3) if j != cls),
            different_old_error=sum(int(t[cls, j, k]) for j in range(3) for k in range(3)
                                    if j != cls and k != cls and j != k))
        assert sum(counts.values()) == support
        for name, count in counts.items():
            rows.append(dict(**prefix, truth=CLASSES[cls], event=name, class_support=support,
                             count=count, class_fraction_pct=100*count/support if support else None))
    return rows


def neighborhoods(cloud, valid):
    boundary = np.zeros(valid.shape, bool)
    vertical = valid[:-1] & valid[1:] & (cloud[:-1] != cloud[1:])
    horizontal = valid[:, :-1] & valid[:, 1:] & (cloud[:, :-1] != cloud[:, 1:])
    boundary[:-1] |= vertical
    boundary[1:] |= vertical
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    h, w = valid.shape
    bp = np.pad(boundary, 2, constant_values=False)
    vp = np.pad(valid, 2, constant_values=False)
    expanded = np.zeros_like(valid)
    interior = np.ones_like(valid)
    for dy in range(5):
        for dx in range(5):
            expanded |= bp[dy:dy+h, dx:dx+w]
            interior &= vp[dy:dy+h, dx:dx+w]
    assert not np.any(interior & ~valid)
    return expanded & interior, ~expanded & interior, valid & ~interior


def test_rules():
    y, a, b = np.indices((3, 3, 3)).reshape(3, -1).astype(np.uint8)
    t = tensor(y, a, b)
    assert np.array_equal(t, np.ones((3, 3, 3), np.int64))
    for row in errors(t, {}):
        assert row['removed'] == row['added'] == 2 and row['corrected_among_removed'] == 1
    same = tensor(y, a, a)
    assert all(r['removed'] == r['added'] == 0 for r in errors(same, {}))
    cloud = np.zeros((11, 11), bool)
    cloud[:, 6:] = True
    valid = np.ones_like(cloud)
    edge, other, excluded = neighborhoods(cloud, valid)
    assert not np.any(edge & other) and np.array_equal(edge | other | excluded, valid)
    assert not edge[:2].any() and not edge[:, :2].any()
    assert int(edge.sum()) == 42 and int(other.sum()) == 7 and int(excluded.sum()) == 72


def catalogue_regions(plan, groups, sealed):
    accum = {region: {arm: np.zeros((3, 3, 3), np.int64) for arm in ('B', 'M')}
             for region in ('cloud_boundary_neighborhood', 'away_from_cloud_boundary')}
    supports, group_rows, region_cells = [], [], []
    with zipfile.ZipFile(DATA/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as archive:
        for row in plan:
            if row['pool'] != 'Catalogue':
                continue
            pid, gid = row['product_id'], row['group']['group_id']
            path = ref.OLD/'infer/R4/scores/Catalogue'/(pid+'.npz')
            assert sha(path) == sealed['R4']['Catalogue/'+pid]
            with np.load(path, allow_pickle=False) as data:
                ids, accepted = data['flat_indices'], data['valid']
            raw = archive.read('masks/'+pid+'.npy')
            import hashlib
            assert hashlib.sha256(raw).hexdigest() == row['group']['mask_member_sha256']
            mask = np.load(io.BytesIO(raw), allow_pickle=False)
            truth = mask.argmax(-1)
            valid = np.zeros(truth.size, bool)
            valid[ids[accepted]] = True
            valid = valid.reshape(truth.shape)
            assert np.all(mask.sum(-1)[valid] == 1)
            g = groups[gid]
            assert np.array_equal(truth.ravel()[ids[accepted]], g['y'])
            edge, away, excluded = neighborhoods(truth == 1, valid)
            from scipy.ndimage import binary_dilation, binary_erosion
            seed = np.zeros_like(valid)
            for axis in (0, 1):
                previous = [slice(None), slice(None)]
                following = [slice(None), slice(None)]
                previous[axis], following[axis] = slice(None, -1), slice(1, None)
                p, q = tuple(previous), tuple(following)
                transition = valid[p] & valid[q] & ((truth[p] == 1) != (truth[q] == 1))
                seed[p] |= transition
                seed[q] |= transition
            independently_eroded = binary_erosion(valid, structure=np.ones((5, 5), bool), border_value=0)
            independently_expanded = binary_dilation(seed, structure=np.ones((5, 5), bool))
            assert np.array_equal(edge, independently_eroded & independently_expanded)
            assert np.array_equal(away, independently_eroded & ~independently_expanded)
            assert np.array_equal(excluded, valid & ~independently_eroded)
            assert np.array_equal(edge | away | excluded, valid) and not (edge & away).any()
            supports.append(dict(product=pid, group=gid, total=int(valid.sum()),
                                 boundary=int(edge.sum()), nonboundary=int(away.sum()), excluded=int(excluded.sum())))
            for region, full in (('cloud_boundary_neighborhood', edge), ('away_from_cloud_boundary', away)):
                selected = full.ravel()[ids[accepted]]
                for arm in ('B', 'M'):
                    t = tensor(g['y'][selected], g['pred']['R4'][selected], g['pred'][arm][selected])
                    accum[region][arm] += t
                    group_rows += errors(t, dict(pool='Catalogue', region=region, product=pid, group=gid, student=arm))
                    region_cells += cells(t, dict(pool='Catalogue', region=region, product=pid, group=gid, student=arm))
    rows = []
    for region, arms in accum.items():
        for arm, t in arms.items():
            key = 'boundary' if region == 'cloud_boundary_neighborhood' else 'nonboundary'
            supported = sum(v[key] > 0 for v in supports)
            rows += errors(t, dict(pool='Catalogue', region=region, student=arm, products=supported, groups=supported, total_groups=32))
    table('TABLE3_BOUNDARY.csv', rows)
    table('BOUNDARY_PER_GROUP.csv', group_rows)
    table('BOUNDARY_SUPPORT.csv', supports)
    table('BOUNDARY_PER_GROUP_TRANSITIONS.csv', region_cells)


def pixbox_strata(plan, sealed):
    global STRATA_CHECKS
    lock = ref.load(P/'SUBGROUP_MAPPING_LOCK.json')
    assert sha(P/'SUBGROUP_MAPPING_LOCK.json') == ref.load(P/'SUBGROUP_ACCEPTANCE.json')['mapping_sha256']
    sha(P/'SUBGROUP_ACCEPTANCE.json')
    refs_path = DATA/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert sha(refs_path) == 'b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'
    refs = {r['point_ids'][0]: r for r in ref.load(refs_path)}
    maps = {arm: {} for arm in ('R4', 'B', 'M')}
    for row in plan:
        if row['pool'] != 'PixBox':
            continue
        pid = row['product_id']
        for arm in maps:
            root = ref.OLD if arm == 'R4' else D
            path = root/'infer'/arm/'scores/PixBox'/pid/'UNLABELLED_POINT_SCORES.json'
            assert sha(path) == sealed[arm]['PixBox/'+pid]
            maps[arm].update({p['point_id']: p for p in ref.load(path)['points']})
    ids = [i for i, v in maps['R4'].items() if v['valid'] and refs[i]['reference'] != 'ignore']
    assert len(ids) == 8569
    codes = {'negative_surface': 0, 'negative_cloud': 1, 'positive_ground_cloud_shadow': 2}
    y = np.array([codes[refs[i]['reference']] for i in ids], np.uint8)
    pred = {arm: np.array([maps[arm][i]['parent_class'] for i in ids], np.uint8) for arm in maps}
    rows, all_cells, supports, by_group = [], [], [], []
    products = np.array([refs[i]['product_id'] for i in ids])
    for pid in sorted(set(products)):
        choose = products == pid
        gids = {refs[ids[i]]['group_id'] for i in np.flatnonzero(choose)}
        assert len(gids) == 1
        for arm in ('B', 'M'):
            t = tensor(y[choose], pred['R4'][choose], pred[arm][choose])
            PRODUCT_CELLS.extend(cells(t, dict(pool='PixBox', student=arm, group=next(iter(gids)), product=pid)))
    for field, mapping in lock['fields'].items():
        field_totals = {arm: np.zeros((3, 3, 3), np.int64) for arm in ('B', 'M')}
        values = []
        for i in ids:
            raw = {str(a.get(field, '')).strip() for a in refs[i]['original_attributes']}
            try:
                value = float(next(iter(raw))) if len(raw) == 1 else float('nan')
                code = str(int(value)) if np.isfinite(value) and value.is_integer() else None
            except (TypeError, ValueError):
                code = None
            values.append(code if code in mapping else None)
        values = np.array(values, object)
        count = 0
        for code, label in list(mapping.items())+[(None, 'missing/conflicting/unmapped')]:
            choose = np.fromiter((v == code for v in values), dtype=bool, count=len(values))
            count += int(choose.sum())
            prefix = dict(pool='PixBox', field=field, code=code, definition=label,
                          products=len({refs[ids[i]]['product_id'] for i in np.flatnonzero(choose)}),
                          groups=len({refs[ids[i]]['group_id'] for i in np.flatnonzero(choose)}),
                          reference_points=int(choose.sum()))
            for arm in ('B', 'M'):
                t = tensor(y[choose], pred['R4'][choose], pred[arm][choose])
                field_totals[arm] += t
                rows += errors(t, dict(**prefix, student=arm))
                all_cells += cells(t, dict(**prefix, student=arm))
                # Per original correlation group, including empty subgroup support.
                for gid in sorted({refs[i]['group_id'] for i in ids}):
                    group_choose = choose & np.array([refs[i]['group_id'] == gid for i in ids])
                    gt = tensor(y[group_choose], pred['R4'][group_choose], pred[arm][group_choose])
                    by_group += errors(gt, dict(pool='PixBox', field=field, code=code, definition=label,
                        group=gid, student=arm, products=len(set(products[group_choose])), reference_points=int(group_choose.sum())))
            supports.append(dict(field=field, code=code, definition=label, reference_points=int(choose.sum()),
                Surface=int((y[choose] == 0).sum()), Cloud=int((y[choose] == 1).sum()), Shadow=int((y[choose] == 2).sum())))
        assert count == 8569
        for arm in ('B', 'M'):
            assert np.array_equal(field_totals[arm], POOLED['PixBox', arm])
            STRATA_CHECKS += 1
    table('TABLE3_PIXBOX_STRATA.csv', rows)
    table('PIXBOX_STRATA_TRANSITIONS.csv', all_cells)
    table('PIXBOX_STRATA_PER_GROUP.csv', by_group)
    table('PIXBOX_STRATA_SUPPORT.csv', supports)


def probes():
    fit = {r['product']: r for r in ref.load(FIT_SOURCE)['bindings']}
    training, full_comparisons, same512_cells = [], [], []
    teacher_lock_path = D/'teacher/CACHE_LOCK.json'
    sha(teacher_lock_path)
    teacher_lock = {r['product']: r for r in ref.load(teacher_lock_path)['rows']}
    with zipfile.ZipFile(DATA/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as archive:
        for arm in ('B', 'M'):
            probe_table = D/('train_'+arm)/'PROBE_TABLE.json'
            sha(probe_table)
            records = ref.load(probe_table)['rows']
            assert len(records) == 16
            assert sorted({v['product'] for v in records}) == sorted(fit)[:4]
            for pid in sorted({v['product'] for v in records}):
                raw = archive.read('masks/'+pid+'.npy')
                import hashlib
                assert hashlib.sha256(raw).hexdigest() == fit[pid]['mask_member_sha256']
                masks = np.load(io.BytesIO(raw), allow_pickle=False)[255:767, 255:767]
                truth = masks.argmax(-1).astype(np.uint8)
                cache_path = D/'teacher'/(pid+'.npz')
                assert sha(cache_path) == teacher_lock[pid]['sha256']
                with np.load(cache_path, allow_pickle=False) as data:
                    sealed_teacher = data['teacher_parent'][255:767, 255:767].copy()
                    sealed_valid = data['valid'][255:767, 255:767].copy()
                initial, common, teacher = None, None, None
                for attempt in (0, 100, 500, 1000):
                    record = next(v for v in records if v['product'] == pid and v['attempt'] == attempt)
                    path = D/('train_'+arm)/'probes'/(pid+'_'+str(attempt)+'.npz')
                    assert sha(path) == record['sha256']
                    with np.load(path, allow_pickle=False) as data:
                        current = data['parent'].copy()
                        valid = data['valid'].copy() & (masks.sum(-1) == 1)
                        teacher_current = data['teacher_parent'].copy()
                    if initial is None:
                        initial, common, teacher = current.copy(), valid, teacher_current
                    assert np.array_equal(valid, common) and np.array_equal(teacher_current, teacher)
                    assert np.array_equal(teacher, sealed_teacher)
                    assert np.array_equal(valid, sealed_valid & (masks.sum(-1) == 1))
                    y, a, b = truth[common], initial[common], current[common]
                    t = tensor(y, a, b)
                    if attempt == 0:
                        assert np.array_equal(a, b)
                    same512_cells += cells(t, dict(product=pid, student=arm, attempt=attempt,
                                                  comparison='own_initial512_to_current512'))
                    for cls in range(3):
                        n = int(t[cls].sum())
                        old_correct, now_correct = int(t[cls, cls].sum()), int(t[cls, :, cls].sum())
                        damaged = old_correct-int(t[cls, cls, cls])
                        repaired = now_correct-int(t[cls, cls, cls])
                        training.append(dict(product=pid, student=arm, attempt=attempt, truth=CLASSES[cls],
                                             class_support=n, initial_correct_current_wrong=damaged,
                                             initial_wrong_current_correct=repaired,
                                             both_wrong=n-int(t[cls, cls, cls])-damaged-repaired,
                                             initial_correct=old_correct, current_correct=now_correct,
                                             net_correct_change=now_correct-old_correct))
                        assert repaired-damaged == now_correct-old_correct
                    if attempt in (0, 1000):
                        ft = tensor(y, teacher[common], b)
                        full_comparisons += cells(ft, dict(product=pid, student=arm, attempt=attempt,
                                                           comparison='full_teacher_to_512_student'))
    table('TABLE4_SAME512_TRAINING.csv', training)
    table('FULL_TEACHER_TO_512.csv', full_comparisons)
    table('SAME512_TRANSITIONS.csv', same512_cells)
    return training


def main():
    started = time.time()
    assert not (O/'CPU_ACCEPTANCE.json').exists() and not (O/'TABLE1_TRANSITIONS.csv').exists(), 'Keep previous deliveries immutable'
    O.mkdir(parents=True, exist_ok=True)
    S.mkdir(parents=True, exist_ok=True)
    ref.sha = sha
    assert 'torch' not in sys.modules
    test_rules()
    protected = R/'src/research_results/phase300_20261020/FINAL_SUMMARY_LOCK.json'
    sha(protected)
    for name, expected_sha in ref.load(protected)['files'].items():
        assert sha(protected.parent/name) == expected_sha, name
    for path in (Path(__file__), R/'src/research_plans/phase301.md', Path(ref.__file__),
                 R/'src/tools/finalize_phase301.py', R/'src/tools/run_phase301.py',
                 D/'TRAINING_SEAL.json', P/'CPU_ACCEPTANCE.json', P/'TABLE1_FIXED_RESULTS.csv',
                 DATA/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip',
                 FIT_SOURCE,
                 R/'src/tools/run_phase300_local.py'):
        sha(path)
    seal = ref.load(D/'TRAINING_SEAL.json')
    for arm in ('B', 'M'):
        assert sha(D/('train_'+arm)/'module_final.safetensors') == seal[arm]
    plan = ref.load(D/'EVALUATION_INPUT_PLAN.json')['rows']
    expected = {v['product_id']: v for v in ref.load(D/'EXPECTED_SUPPORT.json')['rows']}
    sealed = {n: v for n, v in ref.load(ref.OLD/'infer/SCORES_SEALED.json')['scores'].items() if n in ('R4', 'E4')}
    sealed.update(ref.load(D/'infer/SCORES_SEALED.json')['scores'])
    for path in (D/'EVALUATION_INPUT_PLAN.json', D/'EXPECTED_SUPPORT.json',
                 ref.OLD/'infer/SCORES_SEALED.json', D/'infer/SCORES_SEALED.json'):
        sha(path)
    totals, event_rows, per_group, group_errors, state_rows = [], [], [], [], []
    with (P/'TABLE1_FIXED_RESULTS.csv').open(encoding='utf-8-sig') as stream:
        original = {(v['pool'], v['strategy']): np.array(json.loads(v['confusion']), np.int64)
                    for v in csv.DictReader(stream)}
    for pool, reader in (('Catalogue', ref.cat), ('PixBox', ref.pix)):
        groups, _ = reader([v for v in plan if v['pool'] == pool], sealed, expected)
        for arm in ('B', 'M'):
            combined = np.zeros((3, 3, 3), np.int64)
            for gid, g in groups.items():
                t = tensor(g['y'], g['pred']['R4'], g['pred'][arm])
                combined += t
                per_group += cells(t, dict(pool=pool, student=arm, group=gid, products=g['products']))
                group_errors += errors(t, dict(pool=pool, student=arm, group=gid, products=g['products']))
                if pool == 'Catalogue':
                    PRODUCT_CELLS.extend(cells(t, dict(pool=pool, student=arm, group=gid, product=g['products'])))
            assert np.array_equal(combined.sum(2), original[pool, 'R4'])
            assert np.array_equal(combined.sum(1), original[pool, arm+'-PL'])
            totals += cells(combined, dict(pool=pool, student=arm))
            event_rows += errors(combined, dict(pool=pool, student=arm))
            state_rows += states(combined, dict(pool=pool, student=arm))
            POOLED[pool, arm] = combined.copy()
        if pool == 'Catalogue':
            catalogue_regions(plan, groups, sealed)
        print(pool, 'flow identities passed', flush=True)
        del groups
        gc.collect()
    table('TABLE1_TRANSITIONS.csv', totals)
    table('TABLE2_ERROR_EVENTS.csv', event_rows)
    table('PER_GROUP_TRANSITIONS.csv', per_group)
    table('PER_GROUP_ERROR_EVENTS.csv', group_errors)
    table('STATE_EVENTS.csv', state_rows)
    pixbox_strata(plan, sealed)
    table('PER_PRODUCT_TRANSITIONS.csv', PRODUCT_CELLS)
    training = probes()
    if REMOTE:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    else:
        from run_phase300_local import usage
        peak = usage(os.getpid())
    seconds = time.time()-started
    assert peak <= 16*1024**3 and seconds <= 3600
    assert 'torch' not in sys.modules
    write(O/'INPUT_SHA256.json', INPUTS)
    write(O/'CPU_ACCEPTANCE.json', dict(status='passed', neural_calls=0, seconds=seconds,
          peak_bytes=peak, independent_boolean_cell_checks=CHECKS, transition_marginals_match_phase300=True,
          all_original_groups_retained=True, boundary_definition='cloud/noncloud adjacent valid labels; Chebyshev radius2 dilation; exclude invalid/image edge radius2',
          independent_scipy_boundary_checks=32, field_tensor_sum_checks=STRATA_CHECKS,
          input_sha_count=len(INPUTS), torch_imported=False,
          probe_baseline='own initialized adapter at attempt0 with identical512 context; not originalR4'))
    summary = []
    for arm in ('B', 'M'):
        final = [v for v in training if v['student'] == arm and v['attempt'] == 1000]
        summary.append(dict(student=arm, repaired=sum(v['initial_wrong_current_correct'] for v in final),
                            damaged=sum(v['initial_correct_current_wrong'] for v in final),
                            net=sum(v['net_correct_change'] for v in final)))
    write(O/'SUMMARY.json', dict(error_events=event_rows, probe_same512=summary))
    print(json.dumps(dict(seconds=seconds, peak_bytes=peak, same512=summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
