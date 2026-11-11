from pathlib import Path
import json,hashlib,subprocess,ast
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2];L=ROOT/'outputs/phase420/phase420_20261101';RR='/home/scv/shared/phase420_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    L.mkdir(parents=True,exist_ok=True);assert not (L/'LAUNCH.json').exists()
    paths=[ROOT/'src/tools'/n for n in ['run_phase420.py','phase420_shadow.py']]+[ROOT/'src/research_plans/phase420.md']
    for p in paths[:2]:ast.parse(p.read_text(encoding='utf-8'))
    lock={'files':{p.name:sha(p) for p in paths},'seed':40901,'steps_per_arm':1000,'modes':['global','spatial'],'labels_unchanged':True}
    (L/'LOCK.json').write_text(json.dumps(lock,indent=2));paths.append(L/'LOCK.json')
    script=f"""from pathlib import Path
import shutil,hashlib,json
r=Path('{RR}');assert not r.exists();r.mkdir();(r/'code').mkdir()
ref=Path('/home/scv/shared/phase411_20261101/layerwise/code')
for p in ref.glob('*.py'):shutil.copy2(p,r/'code'/p.name)
for n in ['INPUT_LOCK.json','PREPARED_MANIFEST.json']:shutil.copy2(Path('/home/scv/shared/phase400_20261031')/n,r/n)
(r/'BASE_CODE_LOCK.json').write_text(json.dumps({{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').glob('*.py')}}))
print('fresh phase420')
"""
    print(ssh(script).decode())
    subprocess.run(['scp','-o','ConnectTimeout=10',*[str(p) for p in paths],'gzs:'+RR+'/'],check=True,timeout=120)
    script=f"""from pathlib import Path
import subprocess,os,json
r=Path('{RR}')
with (r/'controller.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'run_phase420.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}})
print(json.dumps({{'PID':p.pid,'root':str(r)}}))
"""
    data=ssh(script);(L/'LAUNCH.json').write_bytes(data);print(data.decode())
if __name__=='__main__':main()
