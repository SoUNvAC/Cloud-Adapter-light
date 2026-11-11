from collect_phase241 import ssh
print(ssh("""from pathlib import Path
import subprocess,sys
r=Path('/home/scv/shared/phase494_20261109')
subprocess.run([sys.executable,str(r/'status_shadow494.py')],check=True)
for n in ['POSTPROCESS_COMPLETE.json','POSTPROCESS_FAILURE.json']:
 p=r/n
 if p.exists():print(n,p.read_text())
for seed in [45202,45203]:
 d=r/f'seed_{seed}'/'B/network_run'
 print(seed,'evaluations',[p.name for p in sorted(d.glob('*INDEPENDENT_*.json'))])
""").decode(),flush=True)
