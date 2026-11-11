from pathlib import Path
import subprocess
r=Path('/home/scv/shared/phase501_20261111')
print(subprocess.check_output(['pgrep','-af','postprocess501']).decode())
print((r/'code/tools/postprocess501.py').read_text())
