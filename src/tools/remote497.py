"""Small, explicit remote operations for Phase497 (no checkpoints transferred)."""
import json, sys, tarfile, io, base64
from pathlib import Path
from collect_phase241 import ssh

sys.stdout.reconfigure(encoding="utf-8")

ROOT=Path(__file__).resolve().parents[2]
REMOTE='/home/scv/shared/phase497_20261110'
if sys.argv[1]=='deploy':
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        for p in (ROOT/'src/phase497').glob('*.py'):t.add(p,arcname='phase497/'+p.name)
        t.add(ROOT/'src/research_plans/phase497.md',arcname='phase497.md')
    script=f"""
import base64,io,tarfile,os,json,shutil,torch
from pathlib import Path
r=Path('{REMOTE}'); real=Path('/mnt/d/CloudAdapterExperiments/phase497_20261110')
assert not r.exists() and not real.exists()
assert shutil.disk_usage('/mnt/d').free>50*1024**3
print('GPU',torch.cuda.get_device_name(0),torch.cuda.mem_get_info())
real.mkdir();r.symlink_to(real,target_is_directory=True);(r/'code').mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode('{base64.b64encode(b.getvalue()).decode()}')),mode='r:gz') as t:t.extractall(r/'code')
print('DEPLOYED',r)
"""
elif sys.argv[1]=='status':
    script=f"""
from pathlib import Path
import json
r=Path('{REMOTE}')
for p in [r/'STATUS.json',r/'FAILURE.txt',r/'RESULT.json']:
 if p.exists():print(p.name,p.read_text()[-6000:])
for name in ['preflight_control','preflight_texture','control','texture']:
 p=r/(name+'.log')
 if p.exists(): print(name,p.read_text(errors='replace')[-1600:])
"""
elif sys.argv[1]=='run':
    script=f"""
import subprocess,os
from pathlib import Path
r=Path('{REMOTE}')
env=os.environ.copy();env.update(PYTHONPATH=str(r/'code')+':/home/scv/Cloud-Adapter-light/src',DINOV3_REPO='/home/scv/shared/phase429_20261102/dinov3',DINOV3_WEIGHTS='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth',CLOUDSEN_ROOT='/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
assert not (r/'controller.log').exists()
with (r/'controller.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-m','phase497.controller'],env=env,cwd=r/'code',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('CONTROLLER',p.pid)
"""
else:
    script=Path(sys.argv[1]).read_text(encoding='utf-8')
print(ssh(script).decode())
