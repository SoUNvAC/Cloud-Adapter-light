from pathlib import Path
import subprocess,json,hashlib
from collect_phase241 import ssh
P=Path(__file__).parent
if __name__=='__main__':
    print(ssh("""from pathlib import Path
import shutil
r=Path('/mnt/d/CloudAdapterExperiments/phase444_retrain_20261103');r.mkdir(exist_ok=False)
Path('/home/scv/shared/phase444_retrain_20261103').symlink_to(r,target_is_directory=True)
(r/'code').mkdir()
for p in ['429','431']:shutil.copytree(Path('/home/scv/shared/phase'+p+'_20261102/code')/('phase'+p),r/'code'/('phase'+p),ignore=shutil.ignore_patterns('__pycache__'))
assert not Path('/home/scv/shared/phase444_20261103').exists()
print('Fresh training root ready; old root absent')
""").decode())
    names=['alignment444.py','train_phase444.py','eval_phase444.py','check_phase444.py','report_phase444.py','controller_phase444.py']
    subprocess.run(['scp']+[str(P/n) for n in names]+['gzs:/home/scv/shared/phase444_retrain_20261103/'],check=True)
    print(ssh("""from pathlib import Path
import subprocess,os,json,hashlib
r=Path('/home/scv/shared/phase444_retrain_20261103')
(r/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*.py')},indent=2))
with (r/'run.log').open('w') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u',str(r/'controller_phase444.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(r)+':'+str(r/'code')))
(r/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,models=['baseline429','source431'],iterations_each=40000,target_label_supervision_percent=0)))
print(json.dumps(dict(pid=p.pid)))
""").decode())
