from pathlib import Path
import subprocess
import sys

root=Path('/home/scv/shared/phase438_20261103')
arm=sys.argv[1]
for step in ([0] if arm=='SOURCE' else [4000]+list(range(500,4000,500))):
    if (root/arm/'diagnostics'/str(step)/'COMPLETE.json').exists():continue
    subprocess.run([sys.executable,'-u',str(root/'diagnose_phase438.py'),arm,str(step)],check=True)
