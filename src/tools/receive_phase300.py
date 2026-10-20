"""Back up the entire Phase300 remote preflight directory and console log."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO/'outputs/phase300/phase300_20261020'
REMOTE = '/home/scv/shared/phase300_20261020_backup.tar.gz'


def run(args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    # The marker is written only after all preflight products have completed.
    command = ('test -f /home/scv/shared/phase300_20261020/EXIT_CODE && '
               'tar -czf '+REMOTE+' -C /home/scv/shared '
               'phase300_20261020 phase300_20261020_preflight.console.log && '
               'sha256sum '+REMOTE)
    result = run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'gzs', command])
    expected = result.split()[0]
    archive = LOCAL/'remote_all_results_backup.tar.gz'
    run(['scp', '-o', 'BatchMode=yes', 'gzs:'+REMOTE, str(archive)])
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert actual == expected, 'Remote/local archive SHA mismatch'
    destination = LOCAL/'remote_backup'
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert member.isdir() or member.isfile(), 'Links/devices are forbidden'
            assert path.parts[0] in ('phase300_20261020', 'phase300_20261020_preflight.console.log')
        source.extractall(destination, filter='data')
    files = {str(p.relative_to(destination)):dict(bytes=p.stat().st_size,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in destination.rglob('*') if p.is_file()}
    receipt = dict(status='verified', remote_archive=REMOTE,
                   archive_sha256=actual, files=files, file_count=len(files),
                   scope='Entire Phase300 directory including caches/tmp and external preflight console log')
    (LOCAL/'BACKUP_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='verified', file_count=len(files), archive_sha256=actual)))


if __name__ == '__main__':
    main()
