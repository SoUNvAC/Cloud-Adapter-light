from pathlib import Path
import subprocess,time,json,traceback,sys
R=Path('/home/scv/shared/phase481_20261108')
try:
 start=time.monotonic()
 while True:
  exits=[R/f'{a}_EXIT_CODE' for a in ['C','A','B']]
  for p in exits:
   if p.exists():assert p.read_text().strip()=='0',(p.name,p.read_text())
  assert time.monotonic()-start<14400,'Completion timeout; inspect lane logs'
  if all(p.exists() for p in exits):break
  time.sleep(20)
 for script in ['accept_shadow481.py','summarize_shadow481.py']:
  subprocess.run([sys.executable,str(R/script)],check=True,cwd=R)
 (R/'POSTPROCESS_COMPLETE.json').write_text(json.dumps(dict(passed=True)))
except BaseException:
 (R/'POSTPROCESS_FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())))
 raise
