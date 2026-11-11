"""Bounded, user-authorized read-only SSH inventory; no remote payload writes."""
from pathlib import Path
import base64
import json
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/research_audits/gzs_readonly_intake_20261031'
REMOTE = r'''
from pathlib import Path
import datetime, hashlib, json, os, platform, re, shutil, subprocess

repo=Path('/home/scv/Cloud-Adapter-light')
shared=Path('/home/scv/shared')
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'user':os.environ.get('USER'),'hostname':platform.node(),'kernel':platform.release(),
     'read_only':True,'new_neural_calls':0}
def cmd(args,timeout=15):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,
                         env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
        return {'exit':p.returncode,'stdout':p.stdout[:10000],'stderr':p.stderr[:1500]}
    except Exception as e:return {'error':type(e).__name__+': '+str(e)}
def load(p):return json.loads(p.read_text())
out['uptime']=cmd(['uptime'])
out['memory']=cmd(['free','-h'])
out['gpu']=cmd(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
out['gpu_processes']=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
if 'error' in out['gpu'] and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
    out['gpu']=cmd(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
    out['gpu_processes']=cmd(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
    out['gpu_probe_note']='WSL nvidia-smi is outside noninteractive SSH PATH; used absolute path'
out['disks']={}
for p in [repo,shared,Path('/tmp')]:
    if p.exists():
        u=shutil.disk_usage(p)
        out['disks'][str(p)]={'resolved':str(p.resolve()),'free_GiB':round(u.free/1024**3,2),'total_GiB':round(u.total/1024**3,2)}
hostpowershell=Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
if hostpowershell.is_file():
    out['windows_host_volumes']=cmd([str(hostpowershell),'-NoProfile','-NonInteractive','-Command',
        "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace,Size | ConvertTo-Json -Compress"],timeout=20)
out['instructions']={}
for p in [Path('/AGENTS.md'),Path('/home/AGENTS.md'),Path('/home/scv/AGENTS.md'),repo/'AGENTS.md',repo/'src/AGENTS.md',shared/'AGENTS.md']:
    if p.is_file():out['instructions'][str(p)]=p.read_text(errors='replace')[:14000]
out['repo_git']=cmd(['git','-C',str(repo),'status','--short','--untracked-files=no'])
out['repo_branch']=cmd(['git','-C',str(repo),'branch','--show-current'])
out['repo_head']=cmd(['git','-C',str(repo),'log','-1','--format=%h %s'])
out['data_roots']=[]
data=repo/'src/data'
if data.exists():
    for p in sorted(data.iterdir())[:40]:
        out['data_roots'].append({'name':p.name,'directory':p.is_dir(),'symlink':p.is_symlink(),'resolved':str(p.resolve())})
out['python_runtimes']=[{'path':str(p),'exists':p.exists()} for p in [Path('/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'),Path('/home/scv/miniconda3/bin/python')]]
runtime=Path('/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python')
if runtime.is_file():
    out['torch_runtime']=cmd([str(runtime),'-B','-c',"import sys,json,torch; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}))"],timeout=25)
# Inspect only own visible process metadata; do not print full argv or environment.
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        comm=(p/'comm').read_text().strip()
        if not any(s in comm.lower() for s in ['python','train','controller','jupyter','tmux','screen']):continue
        args=(p/'cmdline').read_bytes().split(b'\x00')
        scripts=[Path(v.decode(errors='replace')).name for v in args if v.endswith(b'.py')]
        status=(p/'status').read_text()
        rss=re.search(r'^VmRSS:\s+(.+)$',status,re.M)
        processes.append({'pid':int(p.name),'comm':comm,'scripts':scripts[:3],
                          'RSS':rss.group(1) if rss else None,'cwd':str((p/'cwd').resolve())})
    except (OSError,PermissionError,ProcessLookupError):continue
out['own_processes']=processes[:60]
out['recent_shared_dirs']=[]
if shared.is_dir():
    dirs=[p for p in shared.iterdir() if p.is_dir()]
    for p in sorted(dirs,key=lambda p:p.stat().st_mtime,reverse=True)[:15]:
        out['recent_shared_dirs'].append(p.name)
out['phases']={}
for name in ['phase241_20261030','phase242_20261031']:
    p=shared/name
    row={'exists':p.is_dir()}
    if p.is_dir():
        for n in ['EXIT_CODE','CONTROLLER_STATUS.json','RUN_STATUS.json','DELIVERY_COMPLETE.json','COMPLETE.json']:
            q=p/n
            if q.is_file():
                if n=='EXIT_CODE':row[n]=q.read_text().strip()
                else:
                    v=load(q);row[n]={k:v[k] for k in ['PID','stage','passed','status','formal_bytes','manifest_SHA','network_calls','cumulative_calls'] if k in v}
        d=p/'delivery'
        if d.is_dir():
            for n in ['INDEPENDENT_ACCEPTANCE.json','COST.json']:
                q=d/n
                if q.is_file():
                    v=load(q);row['delivery/'+n]={k:v[k] for k in ['passed','checks','network_calls','gradient_updates','new_thresholds','new_bootstrap','cumulative_calls','GPU_stage_seconds','CPU_stage_seconds','CPU_seconds','calls','new_bytes'] if k in v}
            manifest=d/'SHA256SUMS.json'
            if manifest.exists():row['delivery_manifest_SHA256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
        if name.startswith('phase241'):
            refs=[]
            for split in ['val','test']:
                q=p/(split.upper()+'_SCORE_INDEX.jsonl')
                if q.exists():
                    rows=[json.loads(l) for l in q.read_text().splitlines() if l]
                    unique={}
                    for r in rows:unique.setdefault((r['domain'],r['model']),r)
                    sample=[]
                    for r in unique.values():
                        score=p/r['score_file'];label=p/r['label_file']
                        sample.append({'domain':r['domain'],'model':r['model'],'score_exists':score.is_file(),
                                       'score_bytes':score.stat().st_size if score.is_file() else None,
                                       'label_exists':label.is_file()})
                    refs.append({'split':split,'index_rows':len(rows),'index_SHA256':hashlib.sha256(q.read_bytes()).hexdigest(),'first_input_per_model_domain':sample})
            row['cache_metadata_sample']=refs
    out['phases'][name]=row
print(json.dumps(out,ensure_ascii=False))
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = base64.b64encode(REMOTE.encode()).decode('ascii')
    command = 'python3 -B -c ' + shlex.quote(
        'import base64;exec(base64.b64decode(' + repr(payload) + ').decode())')
    attempts = []
    for attempt in range(1, 5):
        try:
            p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=5',
                '-o', 'ServerAliveCountMax=2', 'gzs', command],
                capture_output=True, timeout=75)
            attempts.append({'attempt':attempt,'exit':p.returncode,
                             'stderr':p.stderr.decode(errors='replace')[-1800:]})
            if p.returncode == 0:
                value = json.loads(p.stdout.decode('utf-8'))
                (OUT/'SNAPSHOT.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                (OUT/'CONNECTION_ATTEMPTS.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                print(json.dumps(value,ensure_ascii=False,indent=2))
                return
        except subprocess.TimeoutExpired:
            attempts.append({'attempt':attempt,'timeout_seconds':75})
        if attempt < 4:
            time.sleep(3)
    (OUT/'CONNECTION_ATTEMPTS.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    raise SystemExit('Four bounded SSH attempts failed; local retry receipt retained.')


if __name__ == '__main__':
    main()
