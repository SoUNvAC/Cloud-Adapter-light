"""Seal or verify the final Phase301 delivery identically on Windows and Linux."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = repo / 'src/research_results/phase301_20261020'
    out = root / 'verified_run'
    lock_path = root / 'FINAL_VERIFIED_LOCK.json'
    if args.seal:
        assert not lock_path.exists()
        dispatch = json.loads((repo / 'outputs/phase301/phase301_20261020/remote_backup/DISPATCH.json').read_text())
        files = [root / 'phase301.md'] + sorted(p for p in out.rglob('*') if p.is_file())
        code = dict(dispatch['code'])
        for name in ('provenance_phase301.py', 'verify_phase301_summary.py'):
            code['src/tools/' + name] = sha(repo / 'src/tools' / name)
        lock_path.write_text(json.dumps(dict(status='completed',
            files={p.relative_to(root).as_posix(): sha(p) for p in files}, code=code), indent=2) + '\n')
    lock = json.loads(lock_path.read_text())
    assert lock['status'] == 'completed'
    for name, digest in lock['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(root / name) == digest, name
    for name, digest in lock['code'].items():
        assert sha(repo / name) == digest, name
    for name, digest in json.loads((out / 'DELIVERY_SHA256.json').read_text()).items():
        assert sha(out / name) == digest, name
    # Source and Phase300 summary bytes must still match the runtime input lock.
    checked = 0
    prefix = '/home/scv/Cloud-Adapter-light/'
    for name, digest in json.loads((out / 'INPUT_SHA256.json').read_text()).items():
        if name.startswith(prefix):
            assert sha(repo / name[len(prefix):]) == digest, name
            checked += 1
    cpu = json.loads((out / 'CPU_ACCEPTANCE.json').read_text())
    independent = json.loads((out / 'INDEPENDENT_ACCEPTANCE.json').read_text())
    assert cpu['status'] == independent['status'] == 'passed'
    assert cpu['neural_calls'] == independent['neural_calls'] == 0 and not cpu['torch_imported']
    receipt = json.loads((out / 'BACKUP_RECEIPT.json').read_text())
    assert receipt['all_archive_members_verified']
    print(json.dumps(dict(status='passed', summary_files=len(lock['files']),
        code_files=len(lock['code']), runtime_repo_inputs_rechecked=checked,
        backup_sha256=receipt['sha256'], neural_calls=0)))


if __name__ == '__main__':
    main()
