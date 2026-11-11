from collect_phase241 import ssh
from launch_phase443 import script
remaining=script[script.index('controller='):]
print(ssh("from pathlib import Path\nimport subprocess,os,json\nr=Path('/home/scv/shared/phase443_20261103')\n"+remaining).decode())
