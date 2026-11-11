"""Deploy one immutable Phase213 run; safe repeat only returns LAUNCH."""
from pathlib import Path
import hashlib,json,subprocess,time,zipfile
ROOT=Path('/home/scv/shared/phase213_20261021_run04')
PKG=Path('/home/scv/shared/phase213_locked_run04.zip')
EXPECTED='f75499f2716ea48735807e6ded3111055e0232b9283ab7402a62530234f47a0b'
CODE_SHA='6f66b0d3ed0cdbdbc810eba679636f4bd935be1bd83a641d3f8791cdd819a32d'
if (ROOT/'LAUNCH.json').exists():print((ROOT/'LAUNCH.json').read_text());raise SystemExit(0)
assert not ROOT.exists(),'Existing unlaunched root must be inspected, never overwritten'
prior=[]
for n in ['phase213_20261021','phase213_20261021_run02','phase213_20261021_run03']:
    p=ROOT.parent/n;launch=p/'LAUNCH.json'
    if launch.exists():
        pid=json.loads(launch.read_text())['PID'];assert not Path('/proc',str(pid)).exists()
    assert not (p/'evaluation/CALL_LEDGER.jsonl').exists() or (p/'evaluation/CALL_LEDGER.jsonl').stat().st_size==0
    prior.append({'root':str(p),'launched':launch.exists(),'calls':0,'evaluation_directory_exists':(p/'evaluation').exists(),'failure_tail':(p/'error.log').read_text()[-1800:] if (p/'error.log').exists() else 'deployment only; no LAUNCH'})
assert hashlib.sha256(PKG.read_bytes()).hexdigest()==EXPECTED
# Read-only fixed Windows volume free-space metadata, not WSL virtual capacity.
ps='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
r=subprocess.run([ps,'-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],capture_output=True,text=True,timeout=30,check=True)
drives=json.loads(r.stdout.strip());drives=drives if isinstance(drives,list) else [drives]
assert drives and all(int(d['FreeSpace'])>=4*1024**3 for d in drives)
ROOT.mkdir();code=ROOT/'code';code.mkdir()
with zipfile.ZipFile(PKG) as z:
    assert z.testzip() is None and set(z.namelist())=={'execute_phase213_locked.py','ALPHA_SELECTION_LOCK.json','SOURCE_CACHE_HASHES.json','HEAD_HASHES.json'}
    z.extractall(code)
assert hashlib.sha256((code/'execute_phase213_locked.py').read_bytes()).hexdigest()==CODE_SHA
(ROOT/'HOST_DISK.json').write_text(json.dumps({'epoch':time.time(),'fixed_windows_volumes':drives,'minimum_free_bytes':min(int(d['FreeSpace']) for d in drives)},indent=2))
(ROOT/'PRIOR_FAILURES.json').write_text(json.dumps(prior,indent=2))
with (ROOT/'console.log').open('x') as o,(ROOT/'error.log').open('x') as e:
    p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B',str(code/'execute_phase213_locked.py')],cwd='/home/scv/Cloud-Adapter-light/src',stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
receipt={'PID':p.pid,'epoch':time.time(),'expected_calls':11420,'package_sha256':EXPECTED,'code_sha256':CODE_SHA,'stage':'source_recheck_then_seal_then_test'}
(ROOT/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
