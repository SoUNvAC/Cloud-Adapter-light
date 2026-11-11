from pathlib import Path
import subprocess,time,json,traceback,sys
R=Path('/home/scv/shared/phase491_20261108')
try:
 start=time.monotonic()
 while True:
  exits=[R/f'{a}_EXIT_CODE' for a in ['A','B']]
  for p in exits:
   if p.exists():assert p.read_text().strip()=='0',(p.name,p.read_text())
  assert time.monotonic()-start<10800,'Completion timeout'
  if all(p.exists() for p in exits):break
  time.sleep(20)
 for arm in ['A','B']:subprocess.run([sys.executable,str(R/'boundary_region491.py'),arm],check=True,cwd=R)
 for script in ['accept_shadow491.py','compare_shadow491.py']:subprocess.run([sys.executable,str(R/script)],check=True,cwd=R)
 (R/'POSTPROCESS_COMPLETE.json').write_text(json.dumps(dict(passed=True)))
except BaseException:
 (R/'POSTPROCESS_FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())))
 raise
