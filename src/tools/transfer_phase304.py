"""SHA-verified full stream backup: no persistent duplicate archive on either host."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile
from phase304_common import R,ROOT,sha,load,write

LOCAL=R/'outputs/phase304/phase304_20261024'

class HashStream:
    def __init__(self,stream):self.stream=stream;self.h=hashlib.sha256();self.bytes=0
    def write(self,b):
        n=self.stream.write(b);self.h.update(b[:n]);self.bytes+=n;return n
    def read(self,n=-1):
        b=self.stream.read(n);self.h.update(b);self.bytes+=len(b);return b
    def flush(self):self.stream.flush()

def stream():
    assert load(ROOT/'STATUS.json')['status'] in ('completed','failed','waiting_for_gpu')
    assert not (ROOT/'BACKUP_STREAM_RECEIPT.json').exists(),'Never duplicate completed backup'
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('BACKUP_FILES.json','BACKUP_STREAM_RECEIPT.json'))
    size=sum(p.stat().st_size for p in files);assert 2*size+64000000<=6000000000
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    write(ROOT/'BACKUP_FILES.json',manifest)
    out=HashStream(sys.stdout.buffer)
    with tarfile.open(fileobj=out,mode='w|gz',compresslevel=1) as tar:
        tar.add(ROOT/'BACKUP_FILES.json',arcname='BACKUP_FILES.json',recursive=False)
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    out.flush()
    receipt=dict(stream_sha256=out.h.hexdigest(),stream_bytes=out.bytes,files=len(files)+1,raw_bytes=size,
        manifest_sha256=sha(ROOT/'BACKUP_FILES.json'),archive_persisted=False,two_copy_bytes=2*size)
    write(ROOT/'BACKUP_STREAM_RECEIPT.json',receipt);sys.stderr.write(json.dumps(receipt)+'\n')

def receive():
    LOCAL.mkdir(parents=True,exist_ok=True);target=LOCAL/'remote_backup'
    assert not target.exists(),'Preserve existing partial or completed backup'
    target.mkdir()
    command='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase304.py stream'
    p=subprocess.Popen(['ssh','-o','BatchMode=yes','gzs',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    incoming=HashStream(p.stdout);manifest=None;members=0
    try:
        with tarfile.open(fileobj=incoming,mode='r|gz') as tar:
            for member in tar:
                name=PurePosixPath(member.name)
                assert member.isfile() and not name.is_absolute() and '..' not in name.parts
                path=target/str(name);assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
                partial=path.with_name(path.name+'.partial');assert not partial.exists()
                h=hashlib.sha256();n=0
                with partial.open('xb') as dest:
                    src=tar.extractfile(member)
                    for b in iter(lambda:src.read(4*1024**2),b''):dest.write(b);h.update(b);n+=len(b)
                if manifest is None:
                    assert str(name)=='BACKUP_FILES.json';manifest=load(partial)
                else:assert n==manifest[str(name)]['bytes'] and h.hexdigest()==manifest[str(name)]['sha256'],str(name)
                partial.rename(path);members+=1
                if members%25==0:print('verified members',members,flush=True)
        for b in iter(lambda:incoming.read(4*1024**2),b''):pass
        stderr=p.stderr.read().decode('utf-8');code=p.wait();assert code==0,stderr
        receipt=json.loads(stderr.strip().splitlines()[-1])
        assert incoming.bytes==receipt['stream_bytes'] and incoming.h.hexdigest()==receipt['stream_sha256']
        assert sha(target/'BACKUP_FILES.json')==receipt['manifest_sha256'] and members==receipt['files']==len(manifest)+1
        for name,v in manifest.items():assert sha(target/name)==v['sha256'] and (target/name).stat().st_size==v['bytes'],name
        receipt.update(all_archive_members_verified=True,all_stream_members_verified=True)
        write(LOCAL/'BACKUP_RECEIPT.json',receipt);write(target/'BACKUP_STREAM_RECEIPT.json',receipt)
        print(json.dumps(receipt))
    except BaseException:
        import traceback
        p.terminate();write(LOCAL/'BACKUP_FAILURE.json',dict(traceback=traceback.format_exc(),verified_members=members,
            action='preserve partial; investigate before any further transfer'));raise

if __name__=='__main__':{'stream':stream,'receive':receive}[sys.argv[1]]()
