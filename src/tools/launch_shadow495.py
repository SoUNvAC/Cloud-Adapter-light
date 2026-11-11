from pathlib import Path
import subprocess,json,hashlib,sys
R=Path('/home/scv/shared/phase495_20261109')
assert json.loads((R/'UNIT_ACCEPTANCE.json').read_text())['passed']
assert not (R/'LAUNCH.json').exists()
for n,h in json.loads((R/'INITIAL_CODE_SHA.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
jobs={}
for arm in ['A','B']:
 assert not (R/arm).exists()
 with (R/f'{arm}.log').open('w') as h:
  p=subprocess.Popen(['bash',str(R/'run_shadow495.sh'),arm],cwd=R,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
 jobs[arm]=p.pid
with (R/'finish.log').open('w') as h:
 p=subprocess.Popen([sys.executable,str(R/'finish_shadow495.py')],cwd=R,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
jobs['finish']=p.pid
(R/'LAUNCH.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
