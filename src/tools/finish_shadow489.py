from pathlib import Path
import subprocess,time,json,traceback,sys
R=Path('/home/scv/shared/phase489_20261108')
try:
 start=time.monotonic()
 while True:
  exits=[R/f'seed_{s}_EXIT_CODE' for s in [45202,45203]]
  for p in exits:
   if p.exists():assert p.read_text().strip()=='0',(p.name,p.read_text())
  assert time.monotonic()-start<10800,'Completion timeout'
  if all(p.exists() for p in exits):break
  time.sleep(20)
 for script in ['accept_shadow489.py','summarize_shadow489.py']:subprocess.run([sys.executable,str(R/script)],check=True,cwd=R)
 (R/'POSTPROCESS_COMPLETE.json').write_text(json.dumps(dict(passed=True)))
except BaseException:
 (R/'POSTPROCESS_FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())))
 raise
