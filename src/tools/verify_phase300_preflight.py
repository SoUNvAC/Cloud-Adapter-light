"""Independent local CPU verification of admission receipts, not model metrics."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/phase300/phase300_20261020'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    backup = ROOT/'remote_backup'
    remote = backup/'phase300_20261020'
    receipt = read(ROOT/'BACKUP_RECEIPT.json')
    for name, info in receipt['files'].items():
        file = backup/name
        assert file.stat().st_size == info['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == info['sha256']
    manifest = read(ROOT/'FIT_IMAGES.json')['rows']
    images = [json.loads(line) for line in (remote/'IMAGE_ACCESS.jsonl').read_text().splitlines()]
    assert len(images) == len(manifest) == 100
    assert {r['image_member']:r['image_member_sha256'] for r in manifest} == {
        r['member']:r['sha256'] for r in images}
    access = [json.loads(line) for line in (remote/'FILE_ACCESS.jsonl').read_text().splitlines()]
    denied = [r for r in access if r['denied']]
    assert len(denied) == 1 and denied[0]['path'].endswith('/masks.zip')
    windows = read(remote/'FIT_WINDOWS.json')['rows']
    assert len(windows) == 100
    T = sum(len(r['actual_windows']) for r in windows)
    status = read(remote/'REMOTE_PREFLIGHT.json')
    assert T == status['teacher_actual_T'] == 400
    assert status['neural_forward_calls'] == status['optimizer_updates'] == 0
    schedule = read(remote/'CROP_SCHEDULE.json')['rows']
    assert [r['attempt'] for r in schedule] == list(range(1,1001))
    assert all(0 <= r['fit_index'] < 100 and 0 <= r['top'] <= 510 and
               0 <= r['left'] <= 510 and type(r['horizontal_flip']) is bool for r in schedule)
    report = dict(status='passed_preflight_receipts_only', image_members=100,
                  teacher_actual_T=T, schedule_attempts=1000,
                  denied_label_open_attempts=1, successful_label_opens=0,
                  backup_file_count=receipt['file_count'], neural_forward_calls=0,
                  note='No teacher, training, probe, model-freeze or metric acceptance claimed')
    (ROOT/'LOCAL_PREFLIGHT_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
