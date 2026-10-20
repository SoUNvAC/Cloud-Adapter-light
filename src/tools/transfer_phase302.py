"""Archive every new Phase302 artifact and SHA-verify its full local backup."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

R=Path(__file__).resolve().parents[2]
DATE='20261020'
REMOTE=Path('/home/scv/shared/phase302_'+DATE)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def collect():
    assert (REMOTE/'EXIT_CODE').exists(), 'Wait for pipeline termination'
    dispatch=load(REMOTE/'DISPATCH.json')
    process=Path('/proc')/str(dispatch['PID'])/'stat'
    assert not process.exists() or process.read_text().split()[2]=='Z'
    code=REMOTE/'code'
    for n,h in load(REMOTE/'EXECUTION_LOCK.json')['code'].items():
        assert sha(R/n)==h,n
        shutil.copyfile(R/n,code/Path(n).name)
    shutil.copyfile(R/'src/research_plans/phase302.md',code/'phase302.md')
    excluded={'delivery.tar.gz','BACKUP_MANIFEST.json','DELIVERY_RECEIPT.json'}
    files=sorted(p for p in REMOTE.rglob('*') if p.is_file() and p.relative_to(REMOTE).parts[0] not in excluded)
    size=sum(p.stat().st_size for p in files)
    assert size*2<6000000000, 'Archive would exceed remote persistent budget'
    manifest={p.relative_to(REMOTE).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    write(REMOTE/'BACKUP_MANIFEST.json',manifest)
    archive=REMOTE/'delivery.tar.gz'
    assert not archive.exists(), 'Preserve prior archive'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in files+[REMOTE/'BACKUP_MANIFEST.json']:tar.add(p,arcname=p.relative_to(REMOTE).as_posix(),recursive=False)
    receipt=dict(status='complete',bytes=archive.stat().st_size,sha256=sha(archive),
        archived_files=len(files)+1,raw_bytes=size,exit_code=(REMOTE/'EXIT_CODE').read_text().strip())
    write(REMOTE/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(receipt))


def receive():
    root=R/'outputs/phase302'/('phase302_'+DATE);root.mkdir(parents=True,exist_ok=True)
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','gzs',
        'cd /home/scv/Cloud-Adapter-light && /home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B src/tools/transfer_phase302.py collect']
    receipt=json.loads(subprocess.check_output(args,text=True,encoding='utf-8'))
    archive=root/'delivery.tar.gz';assert not archive.exists()
    subprocess.run(['scp','-o','BatchMode=yes','gzs:'+str(REMOTE/'delivery.tar.gz'),str(archive)],check=True)
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    target=root/'remote_backup';assert not target.exists();target.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p=Path(member.name)
            assert not p.is_absolute() and '..' not in p.parts and (member.isfile() or member.isdir())
        tar.extractall(target,filter='data')
    manifest=load(target/'BACKUP_MANIFEST.json')
    for n,v in manifest.items():assert (target/n).stat().st_size==v['bytes'] and sha(target/n)==v['sha256'],n
    receipt.update(all_archive_members_verified=True,local_backup=str(target))
    write(root/'BACKUP_RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__': {'collect':collect,'receive':receive}[sys.argv[1]]()
