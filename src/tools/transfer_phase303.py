"""Snapshot all new Phase303 artifacts and verify every backed-up member."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

R=Path(__file__).resolve().parents[2]
HOST=Path('/home/scv/shared/phase303_20261024')
LOCAL=R/'outputs/phase303/phase303_20261024'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def collect(kind):
    state=load(HOST/'STATUS.json')
    assert state['status'] in ('completed','waiting_for_gpu','failed')
    name='delivery_'+kind;archive=HOST/(name+'.tar.gz');assert not archive.exists()
    files=sorted(p for p in HOST.rglob('*') if p.is_file() and not p.relative_to(HOST).parts[0].startswith('delivery_'))
    size=sum(p.stat().st_size for p in files)
    assert sum(p.stat().st_size for p in HOST.rglob('*') if p.is_file())+size<3000000000
    manifest={p.relative_to(HOST).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    manifest_path=HOST/(name+'_FILES.json');write(manifest_path,manifest)
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in files+[manifest_path]:tar.add(p,arcname=p.relative_to(HOST).as_posix(),recursive=False)
    receipt=dict(status=state['status'],bytes=archive.stat().st_size,sha256=sha(archive),archived_files=len(files)+1,raw_bytes=size)
    write(HOST/(name+'_RECEIPT.json'),receipt);print(json.dumps(receipt))


def receive(kind):
    assert kind in ('cached','final')
    LOCAL.mkdir(parents=True,exist_ok=True)
    command='cd /home/scv/Cloud-Adapter-light && /home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B src/tools/transfer_phase303.py collect '+kind
    receipt=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','gzs',command],text=True,encoding='utf-8'))
    archive=LOCAL/('delivery_'+kind+'.tar.gz');assert not archive.exists()
    subprocess.run(['scp','-o','BatchMode=yes','gzs:'+str(HOST/archive.name),str(archive)],check=True)
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    target=LOCAL/('backup_'+kind);assert not target.exists();target.mkdir()
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.isfile()
        tar.extractall(target,filter='data')
    for n,v in load(target/('delivery_'+kind+'_FILES.json')).items():
        assert (target/n).stat().st_size==v['bytes'] and sha(target/n)==v['sha256'],n
    receipt['all_archive_members_verified']=True
    write(LOCAL/('BACKUP_'+kind.upper()+'.json'),receipt);print(json.dumps(receipt))


if __name__=='__main__':{'collect':collect,'receive':receive}[sys.argv[1]](sys.argv[2])
