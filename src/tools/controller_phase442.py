import subprocess,sys
from pathlib import Path
R=Path('/home/scv/shared/phase442_20261103')
subprocess.run([sys.executable,str(R/'extract_phase442.py'),'official'],check=True)
subprocess.run([sys.executable,str(R/'analyze_phase442.py')],check=True)
