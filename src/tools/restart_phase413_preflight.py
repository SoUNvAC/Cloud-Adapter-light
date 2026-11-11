from pathlib import Path
import json
from collect_phase241 import ssh
L=Path(__file__).resolve().parents[2]/'outputs/phase413/phase413_20261101'
v=json.loads((L/'LAUNCH.json').read_text())
script="""from pathlib import Path
import json,os,signal,time
r=Path('/home/scv/shared/phase413_20261101');destination=r.with_name(r.name+'_attempt0_preflight')
assert not destination.exists()
assert json.loads((r/'STATUS.json').read_text())['mode']=='waiting_for_phase412'
pids=[CONTROLLER]
# Only the waiting controller exists; no GPU worker has been started.
for pid in pids:
 p=Path('/proc')/str(pid)/'cmdline'
 if p.exists():
  command=p.read_bytes().replace(b'\\0',b' ')
  if b'phase413_20261101' in command:
   os.killpg(pid,signal.SIGTERM)
time.sleep(2)
(r/'PROTOCOL_RETIRED.json').write_text(json.dumps({'reason':'Pre-training audit adds explicit no-reference output gate and regression check; no GPU training discarded','pids':pids}))
r.rename(destination)
print(str(destination))
""".replace('CONTROLLER',str(v['PID']))
print(ssh(script).decode())
archive=L.with_name(L.name+'_attempt0_preflight');assert not archive.exists();L.rename(archive)
