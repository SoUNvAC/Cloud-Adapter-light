from collect_phase241 import ssh
print(ssh("""from pathlib import Path
import json,subprocess,sys
r=Path('/home/scv/shared/phase493_20261109')
subprocess.run([sys.executable,str(r/'status_shadow493.py')],check=True)
for name in ['LAUNCH.json','A_EXIT_CODE','B_EXIT_CODE','POSTPROCESS_COMPLETE.json','POSTPROCESS_FAILURE.json']:
 p=r/name
 if p.exists():print(name,p.read_text())
for arm in ['A','B']:
 d=r/arm/'network_run'
 print(arm,'evaluation_files',[p.name for p in sorted(d.glob('*INDEPENDENT_*.json'))])
 p=r/f'{arm}_EXIT_CODE'
 if p.exists() and p.read_text().strip()!='0':print((r/f'{arm}.log').read_text()[-5000:])
""").decode(),flush=True)
