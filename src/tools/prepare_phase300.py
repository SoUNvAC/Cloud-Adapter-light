"""Lock Phase300 fit identities using only NPY headers; no labels or network."""
import argparse
import ast
import datetime
import hashlib
import json
import struct
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    datetime.datetime.strptime(args.date, '%Y%m%d')
    root = REPO / 'outputs/phase300' / ('phase300_' + args.date)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'START_LOCK.json').exists():
        raise SystemExit('Existing run lock; do not silently replace')
    old = REPO / 'outputs/phase208/phase208_20261020'
    fit_path = old / '12201_FIT_INPUT_LOCK.json'
    assert sha(fit_path) == '723fb69ceccb669c66e2885de1a12e48b4a0d5a448133534144dc5dfd62a1d19'
    fit = json.loads(fit_path.read_text(encoding='utf-8-sig'))['bindings']
    assert len(fit) == len({r['group_id'] for r in fit}) == len({r['product'] for r in fit}) == 100
    archive = REPO / 'data/sentinel2_cloud_mask_catalogue_4172871/subscenes.zip'
    rows = []
    with zipfile.ZipFile(archive) as source:
        for original in fit:
            product = original['product']
            assert '/' not in product and '\\' not in product and original['split'] == 'fit'
            member = 'subscenes/' + product + '.npy'
            with source.open(member) as stream:
                assert stream.read(6) == b'\x93NUMPY'
                version = tuple(stream.read(2))
                assert version in ((1, 0), (2, 0))
                width, fmt = (2, '<H') if version == (1, 0) else (4, '<I')
                length = struct.unpack(fmt, stream.read(width))[0]
                header = ast.literal_eval(stream.read(length).decode('latin1').strip())
            assert header['shape'] == (1022, 1022, 13) and header['descr'] == '<f4'
            rows.append(dict(product=product, group_id=original['group_id'],
                             image_archive='/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/subscenes.zip',
                             image_member=member, image_member_sha256=original['image_member_sha256'],
                             shape=list(header['shape'])))
    write(root / 'FIT_IMAGES.json', dict(rows=rows, bands=['B04', 'B03', 'B8A'],
          valid_rule='all three finite, nonzero, != float32(65535/10000)',
          invalid_fill=0, label_fields_retained=False, source_fit_sha256=sha(fit_path)))
    plan = json.loads((old / 'EVALUATION_INPUT_PLAN.json').read_text(encoding='utf-8-sig'))
    evaluation = 2 * sum(len(r['actual_patch_coordinates']) for r in plan['rows'])
    # 1022x1022 permits at most four unique 1000x1000 windows. Actual
    # author get_patch nodata shifts/deduplication must be executed remotely.
    teacher_upper = 400
    forward = teacher_upper + 2000 + evaluation + 32
    write(root / 'PRELIMINARY_BUDGET.json', dict(teacher_actual_T=None,
          teacher_shape_upper_bound=teacher_upper, training_max=2000,
          evaluation=evaluation, probes=32, subtotal_upper=forward,
          construction_acceptance_recovery_available=7000-forward,
          cap=7000, GPU_seconds_cap=14400, GPU_bytes_cap=20*1024**3,
          CPU_bytes_cap=16*1024**3, persistent_bytes_cap=6000000000,
          host_free_bytes_floor=4000000000,
          execution_authorized_by_budget=False,
          pending='Actual author windows T, runtime compatibility, measured time and memory'))
    write(root / 'START_LOCK.json', dict(date_beijing=args.date,
          started_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          protocol_sha256=sha(REPO/'src/research_plans/phase300.md'),
          manifest_sha256=sha(root/'FIT_IMAGES.json'), neural_forward_calls=0,
          status='preflight_pending', local_headers_only=True))
    print(json.dumps(dict(fit_products=len(rows), teacher_upper=teacher_upper,
                         evaluation=evaluation, subtotal_upper=forward)))


if __name__ == '__main__':
    main()
