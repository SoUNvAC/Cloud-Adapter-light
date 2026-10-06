"""Resume the fixed immutable backup through SSH, preserving partial on failure."""
import argparse
import subprocess
from pathlib import Path
from phase65a_freeze import allowed, digest

REMOTE = '/home/scv/Cloud-Adapter-light/src/result_backups/phase65_20261006/all_work_dirs.tar'
SSH = ['ssh', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', 'gzs']
CHUNK_BYTES = 32 * 1024 * 1024


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
    # Bound each connection to 32 MiB: the forwarding link has repeatedly timed
    # out on long streams. Never restart or overwrite a completed prefix.
    while offset < size:
        count = min(CHUNK_BYTES, size - offset)
        command = f'dd if={REMOTE} bs=1048576 skip={offset} count={count} iflag=skip_bytes,count_bytes status=none'
        process = subprocess.Popen(SSH + [command], stdout=subprocess.PIPE)
        received = 0
        try:
            with destination.open('ab') as stream:
                while chunk := process.stdout.read(1024 * 1024):
                    if received + len(chunk) > count:
                        raise RuntimeError('Remote stream exceeds requested segment')
                    stream.write(chunk)
                    received += len(chunk)
            code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        if code != 0 or received != count:
            raise RuntimeError('SSH archive segment incomplete; partial preserved')
        offset += received
        if offset % (1024 * 1024 * 1024) == 0:
            print(f'Archive bytes received: {offset}/{size}', flush=True)
    if destination.stat().st_size != size:
        raise RuntimeError('Archive size mismatch; partial preserved')
    print(f'Archive bytes received: {size}; full SHA and inventory verification still required', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', type=Path, required=True)
    receive(parser.parse_args().partial)
