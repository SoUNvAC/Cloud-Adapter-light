"""Resume the fixed immutable backup through SSH, preserving partial on failure."""
import argparse
import subprocess
from pathlib import Path
from phase65a_freeze import allowed, digest

REMOTE = '/home/scv/Cloud-Adapter-light/src/result_backups/phase65_20261006/all_work_dirs.tar'
SSH = ['ssh', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', 'gzs']


def receive(destination):
    destination = allowed(destination)
    size = int(subprocess.check_output(SSH + [f'stat -c %s {REMOTE}'], text=True).strip())
    offset = destination.stat().st_size if destination.exists() else 0
    if offset > size:
        raise ValueError('Partial larger than remote archive')
    if offset:
        remote_hash = subprocess.check_output(SSH + [f'head -c {offset} {REMOTE} | sha256sum'], text=True).split()[0]
        if digest(destination) != remote_hash:
            raise ValueError('Partial prefix SHA mismatch; no resume')
        print(f'Verified SSH resume prefix: {offset} bytes SHA256={remote_hash}', flush=True)
    if offset == size:
        return
    process = subprocess.Popen(SSH + [f'tail -c +{offset + 1} {REMOTE}'], stdout=subprocess.PIPE)
    try:
        with destination.open('ab') as stream:
            while chunk := process.stdout.read(1024 * 1024):
                stream.write(chunk)
        code = process.wait()
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    if code != 0 or destination.stat().st_size != size:
        raise RuntimeError('SSH archive transfer incomplete; partial preserved')
    print(f'Archive bytes received: {size}; full SHA and inventory verification still required', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', type=Path, required=True)
    receive(parser.parse_args().partial)
