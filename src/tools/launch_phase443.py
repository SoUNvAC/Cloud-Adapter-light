from pathlib import Path
import base64,json,hashlib,subprocess
from collect_phase241 import ssh
P=Path(__file__).parent
names=['appearance_phase443.py','extract_phase443.py','analyze_phase443.py','finish_phase443.py','math_phase441.py']
files={n:base64.b64encode((P/n).read_bytes()).decode() for n in names}
script='''from pathlib import Path
import json,base64,subprocess,os,shutil
r=Path('/mnt/d/CloudAdapterExperiments/phase443_20261103');r.mkdir(exist_ok=False)
link=Path('/home/scv/shared/phase443_20261103');link.symlink_to(r,target_is_directory=True)
b=Path('/home/scv/shared/phase442_20261103')
for n in ['INPUTS.json','COMMON_PCA.npz','SCENE_BOOTSTRAP.npy','SW_DIRECTIONS.npy']:shutil.copyfile(b/n,r/n)
'''
script+='files='+repr(files)+'\n'
script+='''for n,v in files.items():(r/n).write_bytes(base64.b64decode(v))
controller="""import subprocess,sys,os,json,traceback
from pathlib import Path
r=Path('/home/scv/shared/phase443_20261103')
try:
 for n,args in [('appearance_phase443.py',[]),('extract_phase443.py',['official']),('analyze_phase443.py',[]),('finish_phase443.py',[])]:
  (r/'CONTROLLER_STATUS.json').write_text(json.dumps(dict(stage=n,pid=os.getpid())))
  subprocess.run([sys.executable,str(r/n)]+args,check=True)
 (r/'EXIT_CODE').write_text('0')
except BaseException:
 (r/'EXIT_CODE').write_text('1');(r/'FAILURE.txt').write_text(traceback.format_exc());raise
"""
(r/'controller_phase443.py').write_text(controller)
env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
with (r/'run.log').open('w') as log:p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u',str(r/'controller_phase443.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd=r)
(r/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,scope='One fixed appearance control',forwards=567)))
print(json.dumps(dict(pid=p.pid,root=str(r))))
'''
if __name__=='__main__':
    out=Path(__file__).resolve().parents[2]/'outputs/phase443/phase443_20261103';out.mkdir(parents=True,exist_ok=True)
    (out/'CODE_SHA.json').write_text(json.dumps({n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names},indent=2))
    setup,rest=script.split('files=',1)
    print(ssh(setup).decode())
    subprocess.run(['scp']+[str(P/n) for n in names]+['gzs:/home/scv/shared/phase443_20261103/'],check=True)
    remaining=rest[rest.index('controller='):]
    print(ssh("from pathlib import Path\nimport subprocess,os,json\nr=Path('/home/scv/shared/phase443_20261103')\n"+remaining).decode())
