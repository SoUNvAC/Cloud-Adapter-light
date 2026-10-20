"""SHA-check the full remote CPU delivery and copy its verified report locally."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

R=Path(__file__).resolve().parents[2]
ROOT=R/'outputs/phase301/phase301_20261020'
S=R/'src/research_results/phase301_20261020/verified_run'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(args): return subprocess.run(args,check=True,capture_output=True,text=True,encoding='utf-8').stdout


def main():
    command='cd /home/scv/Cloud-Adapter-light && /home/scv/shared/phase111_ocm_runtime/runtime_cuda128/bin/python -B src/tools/collect_phase301.py'
    receipt=json.loads(run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','gzs',command]))
    archive=ROOT/'verified_remote_delivery.tar.gz'
    assert not archive.exists(), 'Do not overwrite a prior received archive'
    run(['scp','-o','BatchMode=yes','gzs:/home/scv/shared/phase301_20261020/delivery.tar.gz',str(archive)])
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    target=ROOT/'remote_backup'
    assert not target.exists();target.mkdir()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            path=Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts and (member.isfile() or member.isdir())
        source.extractall(target,filter='data')
    manifest=json.loads((target/'BACKUP_MANIFEST.json').read_text())
    for name,info in manifest.items():
        assert (target/name).stat().st_size==info['bytes'] and sha(target/name)==info['sha256'],name
    new=target/'verified_run'
    for name,expected in json.loads((new/'DELIVERY_SHA256.json').read_text()).items():
        assert sha(new/name)==expected,name
    assert not S.exists();shutil.copytree(new,S)
    # A convenient local copy in the requested output directory, without disturbing the original delivery.
    assert not (ROOT/'verified_run').exists();shutil.copytree(new,ROOT/'verified_run')
    receipt.update(local_backup=str(target),all_archive_members_verified=True)
    (ROOT/'BACKUP_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (S/'BACKUP_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
