"""Bounded host CPU completion and a separate fully hashed incremental backup."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from transfer_phase302 import R,sha,load,write

HOST=Path('/home/scv/shared/phase302_20261020')
LOCAL=R/'outputs/phase302/phase302_20261020'
ATTEMPT=int(sys.argv[2]) if len(sys.argv)>2 else 1
SUFFIX='' if ATTEMPT==1 else '_attempt'+str(ATTEMPT)
SUBDIR='analysis'+SUFFIX


def host():
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert (HOST/'EXIT_CODE').read_text().strip()=='0'
    out=HOST/('cpu_completion'+SUFFIX);assert not out.exists();out.mkdir()
    for name in ('analyze_phase302.py','complete_phase302_cpu.py','transfer_phase302.py'):
        shutil.copyfile(R/'src/tools'/name,out/name)
    write(out/'START.json',dict(epoch=time.time(),execution='gzs host CPU, cloud-lite-pt210',
        authorization='User explicitly moved CPU analysis to host because local hardware/memory insufficient',
        code={n:sha(out/n) for n in ('analyze_phase302.py','complete_phase302_cpu.py','transfer_phase302.py')},
        endpoint_sha256=sha(HOST/'ENDPOINT_LOCK.json'),neural_calls=0))
    started=time.monotonic();peak=0;minimum=2**63;failure=None
    env=dict(os.environ,PHASE302_REMOTE='1',PHASE302_ANALYSIS_SUBDIR=SUBDIR,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(HOST/'tmp'),
        XDG_CACHE_HOME=str(HOST/'cache'),MPLCONFIGDIR=str(HOST/'cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (out/'console.log').open('xb') as stream:
        process=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/analyze_phase302.py')],cwd=R,env=env,
            stdout=stream,stderr=subprocess.STDOUT)
        while process.poll() is None:
            path=Path('/proc')/str(process.pid)/'status'
            if path.exists():
                values={s.split(':')[0]:s.split(':',1)[1].strip() for s in path.read_text().splitlines() if ':' in s}
                peak=max(peak,int(values.get('VmHWM','0 kB').split()[0])*1024)
            available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            minimum=min(minimum,available)
            if peak>16*1024**3:failure='CPU16GiB'
            if available<4000000000:failure='Host free4GB'
            if shutil.disk_usage(HOST).free<4000000000:failure='Disk free4GB'
            if time.monotonic()-started>3600:failure='CPU analysis one-hour guard'
            if sum(p.stat().st_size for p in HOST.rglob('*') if p.is_file())>6000000000:failure='New persistent6GB'
            if failure:process.terminate();break
            time.sleep(.25)
        code=process.wait()
    resource=dict(status='completed' if code==0 and not failure else 'failed',exit_code=code,failure=failure,
        seconds=time.monotonic()-started,peak_working_set_bytes=peak,minimum_available_physical_bytes=minimum,
        execution='gzs host CPU, cloud-lite-pt210',neural_calls=0)
    write(out/'CPU_ANALYSIS_RESOURCE.json',resource)
    paths=sorted(p for parent in (out,HOST/SUBDIR) if parent.exists() for p in parent.rglob('*') if p.is_file())
    write(out/'FILES.json',{p.relative_to(HOST).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths})
    archive=HOST/('cpu_completion'+SUFFIX+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in paths+[out/'FILES.json']:tar.add(p,arcname=p.relative_to(HOST).as_posix(),recursive=False)
    receipt=dict(bytes=archive.stat().st_size,sha256=sha(archive),status=resource['status'],archived_files=len(paths)+1)
    write(out/'DELIVERY.json',receipt);print(json.dumps(receipt))


def receive():
    command='cat /home/scv/shared/phase302_20261020/cpu_completion'+SUFFIX+'/DELIVERY.json'
    receipt=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','gzs',command],text=True))
    archive=LOCAL/('cpu_completion'+SUFFIX+'.tar.gz');assert not archive.exists()
    subprocess.run(['scp','-o','BatchMode=yes','gzs:'+str(HOST/('cpu_completion'+SUFFIX+'.tar.gz')),str(archive)],check=True)
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    target=LOCAL/('host_cpu_backup'+SUFFIX);assert not target.exists();target.mkdir()
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.isfile()
        tar.extractall(target,filter='data')
    for n,v in load(target/('cpu_completion'+SUFFIX)/'FILES.json').items():
        assert sha(target/n)==v['sha256'] and (target/n).stat().st_size==v['bytes'],n
    receipt['all_archive_members_verified']=True
    write(LOCAL/('CPU_BACKUP_RECEIPT_attempt'+str(ATTEMPT)+'.json'),receipt)
    write(LOCAL/'CPU_BACKUP_RECEIPT.json',receipt)
    shutil.copyfile(target/('cpu_completion'+SUFFIX)/'CPU_ANALYSIS_RESOURCE.json',LOCAL/'CPU_ANALYSIS_RESOURCE.json')
    if receipt['status']=='completed':
        assert not (LOCAL/'analysis').exists();shutil.copytree(target/SUBDIR,LOCAL/'analysis')
    print(json.dumps(receipt))


if __name__=='__main__':{'host':host,'receive':receive}[sys.argv[1]]()
