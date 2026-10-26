"""One immutable archive, state-inspected SSH stream, whole/member SHA; preserve failures."""
import subprocess
import sys
import tarfile
import traceback
from pathlib import PurePosixPath
from phase315_common import *
ARCHIVE='phase315_full_backup.tar.gz'
def remote():
    assert load(ROOT/'STATUS.json')['status']=='completed'
    archive=ROOT/ARCHIVE;receipt=ROOT/'ARCHIVE_RECEIPT.json';manifest=ROOT/'BACKUP_FILES.json';partial=ROOT/(ARCHIVE+'.partial')
    state=dict(archive=archive.exists(),receipt=receipt.exists(),manifest=manifest.exists(),partial=partial.exists())
    if archive.exists():
        assert receipt.exists() and manifest.exists() and not partial.exists();r=load(receipt);assert sha(archive)==r['archive_sha256']
    else:
        assert not any(state.values()),'incomplete archive exists; diagnose, never recollect'
        files=sorted(p for p in ROOT.rglob('*') if p.is_file());raw=sum(p.stat().st_size for p in files);assert 4*raw+100000000<=6000000000
        write(manifest,{p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
        with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
            tar.add(manifest,arcname='BACKUP_FILES.json',recursive=False)
            for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        partial.rename(archive);n=archive.stat().st_size;assert 2*raw+2*n+100000000<=6000000000
        r=dict(archive=ARCHIVE,archive_sha256=sha(archive),archive_bytes=n,files=len(files)+1,raw_bytes=raw,manifest_sha256=sha(manifest),projected_two_hosts_including_archives_and_summary=2*raw+2*n+100000000);write(receipt,r)
    out=sys.stdout.buffer;out.write((json.dumps(dict(receipt=r,remote_state_before=state))+'\n').encode());out.flush()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):out.write(block)
    out.flush()
def receive():
    LOCAL.mkdir(parents=True,exist_ok=True)
    assert not any((LOCAL/n).exists() for n in [ARCHIVE,ARCHIVE+'.partial','remote_backup','BACKUP_RECEIPT.json','BACKUP_FAILURE.json'])
    command='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase315.py remote'
    try:
        p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gzs',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        line=p.stdout.readline()
        if not line:raise RuntimeError('SSH no header; stderr='+p.stderr.read().decode(errors='replace')+' exit='+str(p.wait()))
        header=json.loads(line);r=header['receipt'];write(LOCAL/'REMOTE_ARCHIVE_RECEIPT.json',header);partial=LOCAL/(ARCHIVE+'.partial')
        with partial.open('xb') as f:
            for block in iter(lambda:p.stdout.read(1048576),b''):f.write(block)
        error=p.stderr.read().decode(errors='replace');assert p.wait()==0,error
        assert partial.stat().st_size==r['archive_bytes'] and sha(partial)==r['archive_sha256'];partial.rename(LOCAL/ARCHIVE);target=LOCAL/'remote_backup';target.mkdir();manifest=None;count=0
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
        write(target/'ARCHIVE_RECEIPT.json',r);r.update(all_archive_members_verified=True,archive_SHA_verified_before_extraction=True,all_stream_members_verified=True);write(LOCAL/'BACKUP_RECEIPT.json',r);print(json.dumps(r))
    except BaseException:
        write(LOCAL/'BACKUP_FAILURE.json',dict(traceback=traceback.format_exc(),action='preserve partial/header and remote immutable archive; inspect status before independent recovery; never repeat collect blindly'));raise
if __name__=='__main__':{'remote':remote,'receive':receive}[sys.argv[1]]()
