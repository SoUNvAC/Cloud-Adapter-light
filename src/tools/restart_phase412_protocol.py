from pathlib import Path
import json
from collect_phase241 import ssh
L=Path(__file__).resolve().parents[2]/'outputs/phase412/phase412_20261101'
v=json.loads((L/'LAUNCH.json').read_text())
script="""from pathlib import Path
import json,os,signal,time
r=Path('/home/scv/shared/phase412_20261101');destination=r.with_name(r.name+'_attempt0_val1000')
assert not destination.exists()
pids=[CONTROLLER]
if (r/'STATUS.json').exists():pids.append(json.loads((r/'STATUS.json').read_text())['PID'])
for pid in pids:
 p=Path('/proc')/str(pid)/'cmdline'
 if p.exists():
  command=p.read_bytes().replace(b'\\0',b' ')
  if b'phase412_20261101' in command:
   os.killpg(pid,signal.SIGTERM)
time.sleep(2)
(r/'PROTOCOL_RETIRED.json').write_text(json.dumps({'reason':'1000-only validation differs from Phase411 500-step validation and can shift torch RNG; retain attempted results, restart with matched validation cadence','pids':pids}))
r.rename(destination)
print(str(destination))
""".replace('CONTROLLER',str(v['PID']))
print(ssh(script).decode())
archive=L.with_name(L.name+'_attempt0_val1000');assert not archive.exists();L.rename(archive)
