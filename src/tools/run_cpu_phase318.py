"""Durable independent CPU sequence; all references remain behind ALL_SEALED barrier."""
import subprocess,traceback
from phase318_common import *
def main():
 assert load(ROOT/'ALL_SEALED.json')['status']=='sealed';assert not (ROOT/'CPU_RUN_LOCK.json').exists()
 resource_check();write(ROOT/'CPU_RUN_LOCK.json',dict(pid=os.getpid(),epoch=time.time(),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase318*.py')}));(ROOT/'CPU_code').mkdir()
 for p in (R/'src/tools').glob('*phase318*.py'):shutil.copyfile(p,ROOT/'CPU_code'/p.name)
 stages=[]
 try:
  for name in ['analyze_phase318.py','quality_phase318.py','audit_phase318.py','report_phase318.py']:
   start=time.monotonic();peak=0;minimum=2**63;failure=None;write(ROOT/'STATUS.json',dict(status='CPU_running',stage=name))
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
   with (ROOT/(name+'.log')).open('xb') as f:
    p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/name)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    while p.poll() is None:
     try:
      pp=Path('/proc')/str(p.pid)/'status'
      if pp.exists():peak=max(peak,int(next((s.split()[1] for s in pp.read_text().splitlines() if s.startswith('VmHWM:')),'0'))*1024)
      mem=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024;minimum=min(minimum,mem)
      if sum(r['seconds'] for r in stages)+time.monotonic()-start>7200:failure='CPU statistics twohours'
      if peak>16*1024**3 or mem<4*1024**3 or shutil.disk_usage(ROOT).free<8*1024**3 or budget_bytes()>12000000000:failure='CPU16GiB/freeRAM4GiB/disk8GiB/twohosts12GB'
     except FileNotFoundError:pass
     if failure:p.terminate();break
     time.sleep(1)
    code=p.wait()
   receipt=dict(script=name,exit=code,failure=failure,seconds=time.monotonic()-start,peak_RSS=peak,minimum_host_available=minimum);stages.append(receipt);write(ROOT/'CPU_STAGES.json',stages);assert code==0 and not failure,name+' failed; preserve, diagnose, neverretryblindly'
  seal=load(ROOT/'ALL_SEALED.json');write(ROOT/'STATUS.json',dict(status='completed',forwards=seal['forwards'],gpu_seconds=seal['gpu_seconds'],CPU_stages=stages,resources=resource_check(),newdownloads=0));(ROOT/'EXIT_CODE').write_text('0\n');print('Phase318 all CPU stages passed')
 except BaseException:write(ROOT/'CPU_FAILURE.json',dict(traceback=traceback.format_exc(),stages=stages));(ROOT/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':
 if '--launch' in sys.argv:
  assert load(ROOT/'ALL_SEALED.json')['status']=='sealed' and not (ROOT/'CPU_LAUNCH.json').exists()
  with (ROOT/'CPU_controller.log').open('xb') as f:p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve())],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(ROOT/'CPU_LAUNCH.json',dict(PID=p.pid,epoch=time.time()));print('CPUcontrollerPID',p.pid)
 else:main()
