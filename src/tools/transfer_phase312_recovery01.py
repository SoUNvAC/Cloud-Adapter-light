"""Recover only the missing suffix of the existing frozen archive; retain initial partial/failure."""
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import PurePosixPath
from phase312_common import *
ARCHIVE='phase312_full_backup.tar.gz'

def suffix(offset):
    receipt=load(ROOT/'ARCHIVE_RECEIPT.json');path=ROOT/ARCHIVE
    assert sha(path)==receipt['archive_sha256']=='24afa625e81b94115d02895eaad44b1a5cdd1b4c60b9b6a69805aa41d00da405'
    assert 0<offset<receipt['archive_bytes']==path.stat().st_size
    h=hashlib.sha256()
    with path.open('rb') as f:
        remaining=offset
        while remaining:
            b=f.read(min(4*1024**2,remaining));assert b;h.update(b);remaining-=len(b)
        header=dict(receipt=receipt,offset=offset,prefix_SHA=h.hexdigest(),suffix_bytes=path.stat().st_size-offset)
        sys.stdout.buffer.write((json.dumps(header)+'\n').encode());sys.stdout.buffer.flush()
        for b in iter(lambda:f.read(4*1024**2),b''):sys.stdout.buffer.write(b)
        sys.stdout.buffer.flush()

def receive():
    original=LOCAL/(ARCHIVE+'.partial');dest=LOCAL/(ARCHIVE+'.recovery01.partial')
    assert (LOCAL/'BACKUP_FAILURE.json').exists() and original.exists() and not dest.exists()
    assert not (LOCAL/ARCHIVE).exists() and not (LOCAL/'remote_backup').exists() and not (LOCAL/'BACKUP_RECEIPT.json').exists()
    receipt=load(LOCAL/'REMOTE_ARCHIVE_RECEIPT.json');offset=original.stat().st_size;old_sha=sha(original)
    write(LOCAL/'STALL_DIAGNOSIS.json',dict(original_partial_bytes=offset,original_partial_SHA=old_sha,observations='No byte growth across multiple checks for over six minutes; remote scp PID1761387 waiting,0CPU; local free11444158464 bytes; fresh SSH succeeds. Exact transport root cause unproven.',stopped_only_verified_local_processes=[dict(PID=12472,name='scp',start='2026-10-25T20:04:06+08:00'),dict(PID=37588,name='ssh',start='2026-10-25T20:04:06+08:00')],archive_not_recollected=True,scientific_experiment_not_repeated=True))
    command='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase312_recovery01.py suffix '+str(offset)
    p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gzs',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        header=json.loads(p.stdout.readline());assert header['offset']==offset and header['prefix_SHA']==old_sha and header['receipt']==receipt
        with original.open('rb') as source,dest.open('xb') as f:
            shutil.copyfileobj(source,f,4*1024**2);received=0
            for b in iter(lambda:p.stdout.read(1024**2),b''):f.write(b);received+=len(b)
        stderr=p.stderr.read().decode('utf-8',errors='replace');assert p.wait()==0,stderr
        assert received==header['suffix_bytes'] and dest.stat().st_size==receipt['archive_bytes'] and sha(dest)==receipt['archive_sha256']
        assert sha(original)==old_sha;dest.rename(LOCAL/ARCHIVE);target=LOCAL/'remote_backup';target.mkdir();manifest=None;members=0
        with tarfile.open(LOCAL/ARCHIVE,'r:gz') as tar:
            for m in tar:
                name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in m.name
                path=target/str(name);assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_name(path.name+'.partial');assert not partial.exists()
                with partial.open('xb') as f:shutil.copyfileobj(tar.extractfile(m),f,4*1024**2)
                if manifest is None:
                    assert m.name=='BACKUP_FILES.json' and sha(partial)==receipt['manifest_sha256'];manifest=load(partial)
                else:assert sha(partial)==manifest[m.name]['sha256'] and partial.stat().st_size==manifest[m.name]['bytes'],m.name
                partial.rename(path);members+=1
        assert members==receipt['files']==len(manifest)+1
        for n,v in manifest.items():assert sha(target/n)==v['sha256'] and (target/n).stat().st_size==v['bytes'],n
        write(target/'ARCHIVE_RECEIPT.json',receipt)
        recovery=dict(status='passed',original_partial_bytes=offset,original_partial_SHA=old_sha,remote_prefix_SHA_matched=True,only_missing_suffix_transferred_bytes=received,original_partial_preserved=True,original_failure_preserved=True,archive_recollected=False,full_archive_SHA=receipt['archive_sha256'],all_members_verified=members,scientific_fits_or_forwards_added=0,code_SHA=sha(R/'src/tools/transfer_phase312_recovery01.py'))
        write(LOCAL/'BACKUP_RECOVERY01.json',recovery)
        receipt.update(all_archive_members_verified=True,archive_SHA_verified_before_extraction=True,all_stream_members_verified=True,original_SCP_stalled_failure_preserved=True,prefix_verified_suffix_recovery=True)
        write(LOCAL/'BACKUP_RECEIPT.json',receipt);print(json.dumps(recovery))
    except BaseException:
        p.terminate()
        try:p.communicate(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.communicate()
        write(LOCAL/'BACKUP_FAILURE_RECOVERY01.json',dict(traceback=traceback.format_exc(),action='Preserve original and recovery partial; investigate before further action'));raise

if __name__=='__main__':suffix(int(sys.argv[2])) if sys.argv[1]=='suffix' else receive()
