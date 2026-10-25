"""Bounded immutable archive, full SHA before ordinary seekable extraction; no overwrite."""
import subprocess
import sys
import tarfile
import traceback
from pathlib import PurePosixPath
from phase311_common import *

ARCHIVE='phase311_full_backup.tar.gz'
COMMAND='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase311.py collect'

def collect():
    assert load(ROOT/'STATUS.json')['status'] in ('completed','failed')
    assert not (ROOT/ARCHIVE).exists() and not (ROOT/(ARCHIVE+'.partial')).exists()
    assert not (ROOT/'BACKUP_FILES.json').exists() and not (ROOT/'ARCHIVE_RECEIPT.json').exists()
    files=sorted(p for p in ROOT.rglob('*') if p.is_file())
    raw=sum(p.stat().st_size for p in files);assert 4*raw+64000000<=4000000000
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    write(ROOT/'BACKUP_FILES.json',manifest)
    partial=ROOT/(ARCHIVE+'.partial')
    with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
        tar.add(ROOT/'BACKUP_FILES.json',arcname='BACKUP_FILES.json',recursive=False)
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    partial.rename(ROOT/ARCHIVE);n=(ROOT/ARCHIVE).stat().st_size
    assert 2*raw+2*n+64000000<=4000000000
    receipt=dict(archive=ARCHIVE,archive_sha256=sha(ROOT/ARCHIVE),archive_bytes=n,files=len(files)+1,raw_bytes=raw,manifest_sha256=sha(ROOT/'BACKUP_FILES.json'),archive_persisted=True,projected_two_hosts_including_archives_and_summary=2*raw+2*n+64000000)
    write(ROOT/'ARCHIVE_RECEIPT.json',receipt);print(json.dumps(receipt))

def receive():
    LOCAL.mkdir(parents=True,exist_ok=True)
    assert not (LOCAL/'remote_backup').exists() and not (LOCAL/'BACKUP_RECEIPT.json').exists()
    assert not (LOCAL/ARCHIVE).exists() and not (LOCAL/(ARCHIVE+'.partial')).exists()
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','gzs',COMMAND],check=True,capture_output=True,text=True)
        receipt=json.loads(result.stdout.strip().splitlines()[-1]);write(LOCAL/'REMOTE_ARCHIVE_RECEIPT.json',receipt)
        partial=LOCAL/(ARCHIVE+'.partial')
        subprocess.run(['scp','-q','gzs:'+str(ROOT/ARCHIVE),str(partial)],check=True)
        assert partial.stat().st_size==receipt['archive_bytes'] and sha(partial)==receipt['archive_sha256']
        partial.rename(LOCAL/ARCHIVE);target=LOCAL/'remote_backup';target.mkdir();members=0;manifest=None
        with tarfile.open(LOCAL/ARCHIVE,'r:gz') as tar:
            for m in tar:
                name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in m.name
                path=target/str(name);assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_name(path.name+'.partial');assert not temp.exists()
                with temp.open('xb') as dest:
                    stream=tar.extractfile(m)
                    for block in iter(lambda:stream.read(4*1024**2),b''):dest.write(block)
                if manifest is None:
                    assert m.name=='BACKUP_FILES.json' and sha(temp)==receipt['manifest_sha256'];manifest=load(temp)
                else:assert temp.stat().st_size==manifest[m.name]['bytes'] and sha(temp)==manifest[m.name]['sha256'],m.name
                temp.rename(path);members+=1
        assert members==receipt['files']==len(manifest)+1
        for n,v in manifest.items():assert sha(target/n)==v['sha256'] and (target/n).stat().st_size==v['bytes'],n
        write(target/'ARCHIVE_RECEIPT.json',receipt)
        receipt.update(all_archive_members_verified=True,archive_SHA_verified_before_extraction=True,all_stream_members_verified=True)
        write(LOCAL/'BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt))
    except BaseException:
        write(LOCAL/'BACKUP_FAILURE.json',dict(traceback=traceback.format_exc(),action='Preserve archive, partial and any verified members; diagnose before recovery. Never repeat collect/receive blindly.'));raise

if __name__=='__main__':{'collect':collect,'receive':receive}[sys.argv[1]]()
