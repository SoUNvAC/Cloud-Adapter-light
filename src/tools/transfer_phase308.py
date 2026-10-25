"""Small complete Phase308 archive and safe per-file SHA backup; stdlib only."""
import json
import subprocess
import sys
import tarfile
from phase308_common import *

def collect():
    assert load(ROOT/'STATUS.json')['status'] in ('completed','failed')
    target=ROOT/'delivery.tar.gz';assert not target.exists()
    files=sorted(p for p in ROOT.rglob('*') if p.is_file())
    size=sum(p.stat().st_size for p in files);assert 5*size+10000000<1000000000
    write(ROOT/'BACKUP_FILES.json',{p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with tarfile.open(target,'w:gz',compresslevel=1) as tar:
        for p in files+[ROOT/'BACKUP_FILES.json']:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    receipt=dict(bytes=target.stat().st_size,sha256=sha(target),files=len(files)+1,raw_bytes=size)
    write(ROOT/'ARCHIVE_RECEIPT.json',receipt);print(json.dumps(receipt))

def receive():
    LOCAL.mkdir(parents=True,exist_ok=True)
    assert not (LOCAL/'delivery.tar.gz').exists() and not (LOCAL/'remote_backup').exists()
    command='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase308.py collect'
    receipt=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','gzs',command],text=True,encoding='utf-8'))
    archive=LOCAL/'delivery.tar.gz'
    subprocess.run(['scp','-o','BatchMode=yes','gzs:'+str(ROOT/archive.name),str(archive)],check=True)
    assert sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    out=LOCAL/'remote_backup';out.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        tar.extractall(out,filter='data')
    for n,v in load(out/'BACKUP_FILES.json').items():assert sha(out/n)==v['sha256'] and (out/n).stat().st_size==v['bytes'],n
    receipt['all_members_SHA_verified']=True;write(LOCAL/'BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':{'collect':collect,'receive':receive}[sys.argv[1]]()
