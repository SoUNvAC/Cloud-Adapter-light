from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parents[2]/'outputs/phase481/phase481_20261108'
a=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text())
for run,v in a['runs'].items():
 seed,arm=run.split('/');d=R/f'seed_{seed}'/arm/'network_run';d.mkdir(parents=True,exist_ok=True)
 for name,w in v['weights'].items():
  if name=='branch_0.pth':continue
  p=d/name
  if not p.exists():
   remote=f'gzs:/home/scv/shared/phase481_20261108/seed_{seed}/{arm}/network_run/{name}'
   subprocess.run(['scp',remote,str(p)],check=True)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==w['sha256']
 print(run,'weights verified',flush=True)
