"""One immutable metadata backup; SHA before extraction and every member checked."""
import subprocess
import tarfile
import traceback
from pathlib import PurePosixPath
from phase316_common import *
ARCHIVE='phase316_full_backup.tar.gz'
def remote():
    assert load(ROOT/'STATUS.json')['status']=='completed'
    archive=ROOT/ARCHIVE;receipt=ROOT/'ARCHIVE_RECEIPT.json';manifest=ROOT/'BACKUP_FILES.json';partial=ROOT/(ARCHIVE+'.partial')
    if archive.exists():
        r=load(receipt);assert manifest.exists() and not partial.exists() and sha(archive)==r['archive_sha256']
    else:
        assert not any(p.exists() for p in [receipt,manifest,partial]),'preserve incomplete backup and diagnose'
        files=sorted(p for p in ROOT.rglob('*') if p.is_file());raw=sum(p.stat().st_size for p in files)
        assert 4*raw+100000000<=300000000
        write(manifest,{p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
        with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
            tar.add(manifest,arcname='BACKUP_FILES.json',recursive=False)
            for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        partial.rename(archive);budget=2*raw+2*archive.stat().st_size+100000000;assert budget<=300000000
        r=dict(archive=ARCHIVE,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(manifest),files=len(files)+1,raw_bytes=raw,projected_persistent_bytes=budget);write(receipt,r)
    sys.stdout.buffer.write((json.dumps(r)+'\n').encode());sys.stdout.buffer.flush()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(block)
    sys.stdout.buffer.flush()
def receive():
    LOCAL.mkdir(parents=True,exist_ok=True)
    assert not any((LOCAL/n).exists() for n in [ARCHIVE,ARCHIVE+'.partial','remote_backup','BACKUP_RECEIPT.json','BACKUP_FAILURE.json'])
    try:
        command='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase316.py remote'
        p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gzs',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        line=p.stdout.readline()
        if not line:raise RuntimeError('SSH noheader: '+p.stderr.read().decode(errors='replace')+'; exit='+str(p.wait()))
        r=json.loads(line);write(LOCAL/'REMOTE_ARCHIVE_RECEIPT.json',r);partial=LOCAL/(ARCHIVE+'.partial')
        with partial.open('xb') as f:
            for block in iter(lambda:p.stdout.read(1048576),b''):f.write(block)
        error=p.stderr.read().decode(errors='replace');assert p.wait()==0,error
        assert partial.stat().st_size==r['archive_bytes'] and sha(partial)==r['archive_sha256'];partial.rename(LOCAL/ARCHIVE)
        target=LOCAL/'remote_backup';target.mkdir();manifest=None;count=0
        with tarfile.open(LOCAL/ARCHIVE,'r:gz') as tar:
            for m in tar:
                name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in m.name
                path=target/str(name);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial');assert not path.exists()
                with temp.open('xb') as f:
                    stream=tar.extractfile(m)
                    for block in iter(lambda:stream.read(1048576),b''):f.write(block)
                if manifest is None:assert m.name=='BACKUP_FILES.json' and sha(temp)==r['manifest_sha256'];manifest=load(temp)
                else:assert sha(temp)==manifest[m.name]['sha256'] and temp.stat().st_size==manifest[m.name]['bytes'],m.name
                temp.rename(path);count+=1
        assert count==r['files']==len(manifest)+1
        r.update(all_members_SHA_verified=True,archive_SHA_verified=True);write(target/'ARCHIVE_RECEIPT.json',r);write(LOCAL/'BACKUP_RECEIPT.json',r);print(json.dumps(r))
    except BaseException:
        write(LOCAL/'BACKUP_FAILURE.json',dict(traceback=traceback.format_exc(),action='preserve partial/receipt and diagnose before any recovery; never recollect blindly'));raise
if __name__=='__main__':{'remote':remote,'receive':receive}[sys.argv[1]]()
