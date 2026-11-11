from pathlib import Path
import subprocess,json,time
r=Path('/home/scv/shared/phase497_20261110')
print('uptime',Path('/proc/uptime').read_text())
print(subprocess.check_output(['ps','-eo','pid,lstart,args'],text=True)[-1000:] if False else subprocess.getoutput("ps -eo pid,lstart,args | grep -E 'phase497|python.*train|frpc' | grep -v grep"))
print('checkpoints',[(p.name,p.stat().st_size) for p in (r/'texture').glob('*.pth')])
print('controller_tail',(r/'controller.log').read_text(errors='replace')[-3000:])
print('now',time.time(),'log_mtime',(r/'texture.log').stat().st_mtime)
print(subprocess.getoutput('/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'))
