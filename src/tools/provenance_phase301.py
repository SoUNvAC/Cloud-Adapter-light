"""Append actual remote execution provenance without rewriting sealed CPU results."""
import hashlib
import json
from pathlib import Path
import shutil

R = Path('/home/scv/Cloud-Adapter-light')
ROOT = Path('/home/scv/shared/phase301_20261020')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = ROOT / 'verified_run'
    assert (ROOT / 'EXIT_CODE').read_text().strip() == '0'
    for name, digest in json.loads((out / 'DELIVERY_SHA256.json').read_text()).items():
        assert sha(out / name) == digest, name
    checks = json.loads((out / 'INDEPENDENT_ACCEPTANCE.json').read_text())
    assert checks['status'] == 'passed' and checks['neural_calls'] == 0
    record = dict(
        status='verified', host='ssh gzs', execution='CPU only',
        runtime='/home/scv/shared/phase111_ocm_runtime/runtime_cuda128/bin/python',
        network='Local git push, remote git pull; SCP of existing PixBox reference fixture; SCP of full delivery back to local workspace.',
        inputs='Sealed Phase300 scores and probes in /home/scv/shared; existing reference and mask fixtures; no new dataset or weight downloads.',
        clarification='Supersedes the generic local-backup/network sentence in EXECUTION_NOTES.json. That sealed file is retained byte for byte.',
        training=0, neural_calls=0, external_downloads=0,
        source_sha256=sha(Path(__file__)),
        sealed_delivery_manifest_sha256=sha(out / 'DELIVERY_SHA256.json'))
    target = out / 'RUNTIME_PROVENANCE.json'
    assert not target.exists()
    target.write_text(json.dumps(record, indent=2) + '\n')
    (ROOT / 'code').mkdir(exist_ok=True)
    shutil.copyfile(__file__, ROOT / 'code' / Path(__file__).name)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
