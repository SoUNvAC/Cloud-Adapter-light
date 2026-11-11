"""One-shot bounded completion/receive chain, no scheduled task or GPU retry."""
from pathlib import Path,PurePosixPath
import subprocess,json,hashlib,os,sys,tarfile,time,traceback
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/tools').is_dir());OUT=ROOT/'outputs/phase227/phase227_20261025'
SSH=['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs']
REMOTE="""
import os,json,time,subprocess,hashlib,sys
from pathlib import Path
r=Path('/home/scv/shared/phase227_20261025');deadline=time.monotonic()+21660
while not (r/'EXIT_CODE').exists():
    try:os.kill(int((r/'PID').read_text()),0)
    except ProcessLookupError:raise RuntimeError('Worker exited without EXIT_CODE; preserve failure, no restart')
    assert time.monotonic()<deadline,'bounded completion wait exceeded'
    time.sleep(30)
assert (r/'EXIT_CODE').read_text().strip()=='0',(r/'CONTROLLER_FAILURE.txt').read_text() if (r/'CONTROLLER_FAILURE.txt').exists() else 'worker resource failure'
archive=r.parent/'phase227_delivery.tar.gz'
if not archive.exists():
    proc=subprocess.run(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B',str(r/'code/collect_phase227.py')],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stdout+'\\n'+proc.stderr)
    receipt=json.loads(proc.stdout.strip().splitlines()[-1])
else:
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'SHA':h.hexdigest(),'remote_scope_bytes':sum(p.stat().st_size for p in r.rglob('*') if p.is_file()),'existing_archive_reused':True}
print(json.dumps(receipt),flush=True)
"""
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    assert not (OUT/'RECEIVER_RECOVERY02_STARTED.json').exists(),'One receiver only; diagnose exited receiver before recovery'
    save('RECEIVER_RECOVERY02_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'kind':'one-shot bounded wait and delivery, not an automation','new_model_calls':0})
    save('LOCAL_STATUS.json',{'stage':'waiting_existing_remote_worker','PID':os.getpid()})
    result=None
    for attempt in range(1,5):
        proc=subprocess.run(SSH+['python3 -'],input=REMOTE,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (OUT/f'RECEIVER_SSH_{attempt}.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr,encoding='utf-8')
        if proc.returncode==0:result=json.loads(proc.stdout.strip().splitlines()[-1]);break
        if proc.returncode!=255:raise RuntimeError('Remote completion/collection failed; no model retry: '+proc.stderr[-3000:])
        if attempt<4:time.sleep(5)
    assert result is not None,'SSH failed at most4 attempts'
    save('REMOTE_DELIVERY_RECEIPT.json',result)
    partial=OUT/'phase227_delivery.tar.gz.partial';archive=OUT/'phase227_delivery.tar.gz';assert not partial.exists() and not archive.exists()
    save('LOCAL_STATUS.json',{'stage':'receiving_small_shared_delivery','PID':os.getpid(),'expected_bytes':result['bytes']})
    subprocess.run(['scp','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs:'+result['archive'],str(partial)],check=True)
    assert partial.stat().st_size==result['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==result['SHA'];partial.replace(archive)
    delivery=OUT/'delivery';delivery.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar.getmembers():
            name=PurePosixPath(item.name);assert item.isfile() and not name.is_absolute() and '..' not in name.parts and (not item.name.endswith('.pth') or (item.name=='anchor_head.pth' or (len(name.parts)==4 and name.parts[0] in ['SPARCS','L8'] and name.parts[1] in ['seed21701','seed21702'] and name.parts[2]=='heads' and name.parts[3] in ['W1_iter4000.pth','W10_iter4000.pth'])))
            dest=(delivery/Path(*name.parts)).resolve();assert delivery.resolve() in dest.parents
            dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(item) as src,dest.open('xb') as dst:dst.write(src.read())
    assert all(hashlib.sha256((OUT/'local_code'/n).read_bytes()).hexdigest()==h for n,h in json.loads((OUT/'POSTPROCESS_CODE_LOCK.json').read_text()).items())
    save('LOCAL_STATUS.json',{'stage':'CPU_tables_and_independent_acceptance','PID':os.getpid()})
    for script in ['analyze_phase227.py','validate_phase227.py','finalize_phase227.py']:
        with (OUT/(script+'.log')).open('x',encoding='utf-8') as log:subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'local_code'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    (OUT/'LOCAL_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'LOCAL_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8');save('LOCAL_STATUS.json',{'stage':'failed','PID':os.getpid()});(OUT/'LOCAL_EXIT_CODE').write_text('1\n');raise
