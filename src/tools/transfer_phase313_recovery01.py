"""State-checked backup after initial SSH255; reuse any sealed archive, never overwrite."""
import contextlib
import io
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import PurePosixPath
from phase313_common import *
from transfer_phase313 import ARCHIVE,collect

def remote():
    archive=ROOT/ARCHIVE;receipt=ROOT/'ARCHIVE_RECEIPT.json';manifest=ROOT/'BACKUP_FILES.json';partial=ROOT/(ARCHIVE+'.partial')
    state=dict(archive=archive.exists(),receipt=receipt.exists(),manifest=manifest.exists(),partial=partial.exists())
    assert load(ROOT/'STATUS.json')['status']=='completed'
    if archive.exists():
        assert receipt.exists() and manifest.exists() and not partial.exists()
        r=load(receipt);assert sha(archive)==r['archive_sha256'];action='reuse existing immutable archive; collect not repeated'
    else:
        assert not any(state.values()),'incomplete prior archive; preserve and diagnose, no recollect'
        failure=S/'BACKUP_FAILURE.json';assert failure.is_file()
        shutil.copyfile(failure,ROOT/'BACKUP_FAILURE_FIRST.json')
        write(ROOT/'BACKUP_RECOVERY01.json',dict(initial_receiver_exit=255,initial_receiver_failure_sha256=sha(failure),remote_state_before_recovery=state,diagnosis='No archive, manifest, receipt or archive partial exists; prior collect did not publish any artifact. Underlying initial SSH255 cause unproven.',code_sha256=sha(Path(__file__)),action='one first successful collect, stream same immutable archive; no scientific repeat'))
        with (ROOT/'phase313.md').open('a',encoding='utf-8') as f:f.write('\n- 备份故障：首次SSH子进程返回255，原失败保留；远端四项归档状态均不存在后才独立收尾。原因未证实；不重跑资产核查，首次成功归档经独立SSH流回传，整包及成员SHA验收。\n')
        with contextlib.redirect_stdout(io.StringIO()):collect()
        r=load(receipt);action='clean remote state proved; first successful collect only'
    out=sys.stdout.buffer;out.write((json.dumps(dict(receipt=r,state_before=state,action=action))+'\n').encode());out.flush()
    with archive.open('rb') as src:
        for block in iter(lambda:src.read(1048576),b''):out.write(block)
    out.flush()

def receive():
    assert (LOCAL/'BACKUP_FAILURE.json').exists()
    assert not any((LOCAL/n).exists() for n in [ARCHIVE,ARCHIVE+'.partial','remote_backup','BACKUP_RECEIPT.json'])
    command='source /home/scv/miniconda3/etc/profile.d/conda.sh && conda activate cloud-lite-pt210 && cd /home/scv/Cloud-Adapter-light && python -B src/tools/transfer_phase313_recovery01.py remote'
    process=None
    try:
        process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gzs',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        line=process.stdout.readline()
        if not line:
            error=process.stderr.read().decode(errors='replace');code=process.wait();raise RuntimeError(f'SSH no header, exit={code}, stderr={error}')
        header=json.loads(line);r=header['receipt'];write(LOCAL/'BACKUP_RECOVERY01_HEADER.json',header)
        partial=LOCAL/(ARCHIVE+'.partial')
        with partial.open('xb') as dest:
            for block in iter(lambda:process.stdout.read(1048576),b''):dest.write(block)
        error=process.stderr.read().decode(errors='replace');assert process.wait()==0,error
        assert partial.stat().st_size==r['archive_bytes'] and sha(partial)==r['archive_sha256']
        partial.rename(LOCAL/ARCHIVE);target=LOCAL/'remote_backup';target.mkdir();manifest=None;members=0
        with tarfile.open(LOCAL/ARCHIVE,'r:gz') as tar:
            for member in tar:
                name=PurePosixPath(member.name);assert member.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in member.name
                p=target/str(name);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.partial');assert not p.exists()
                with temp.open('xb') as dest:
                    stream=tar.extractfile(member)
                    for block in iter(lambda:stream.read(1048576),b''):dest.write(block)
                if manifest is None:
                    assert member.name=='BACKUP_FILES.json' and sha(temp)==r['manifest_sha256'];manifest=load(temp)
                else:assert temp.stat().st_size==manifest[member.name]['bytes'] and sha(temp)==manifest[member.name]['sha256'],member.name
                temp.rename(p);members+=1
        assert members==r['files']==len(manifest)+1
        for name,v in manifest.items():assert sha(target/name)==v['sha256'],name
        write(target/'ARCHIVE_RECEIPT.json',r)
        r.update(all_archive_members_verified=True,archive_SHA_verified_before_extraction=True,all_stream_members_verified=True,recovery='state-checked independent SSH stream; no repeated scientific audit or existing archive overwrite')
        write(LOCAL/'BACKUP_RECEIPT.json',r);print(json.dumps(r))
    except BaseException:
        write(LOCAL/'BACKUP_FAILURE_RECOVERY01.json',dict(traceback=traceback.format_exc(),action='preserve partial/header/remote immutable archive; inspect before any action'));raise
if __name__=='__main__':{'remote':remote,'receive':receive}[sys.argv[1]]()
