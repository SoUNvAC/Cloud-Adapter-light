from pathlib import Path
import json,hashlib,subprocess,ast
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2]
L=ROOT/'outputs/phase419/phase419_20261101'
RR='/home/scv/shared/phase419_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    L.mkdir(parents=True,exist_ok=True);assert not (L/'LAUNCH.json').exists()
    paths=[ROOT/'src/tools/diagnose_phase419.py',ROOT/'src/research_plans/phase419.md']
    ast.parse(paths[0].read_text(encoding='utf-8'))
    refs={}
    for phase in [411,416,417,418]:
        p=ROOT/f'outputs/phase{phase}/phase{phase}_20261101/TARGET_SUMMARY.csv'
        refs[f'/home/scv/shared/phase{phase}_20261101/TARGET_SUMMARY.csv']=sha(p)
    refs['/home/scv/shared/phase400_20261031/INPUT_LOCK.json']=sha(ROOT/'outputs/phase400/phase400_20261031/delivery/INPUT_LOCK.json')
    lock={'files':{p.name:sha(p) for p in paths},'references':refs,'new_training_updates':0,'new_model_forwards':0}
    (L/'LOCK.json').write_text(json.dumps(lock,indent=2));paths.append(L/'LOCK.json')
    print(ssh(f"from pathlib import Path\nr=Path('{RR}');assert not r.exists();r.mkdir();print('fresh phase419')").decode())
    subprocess.run(['scp','-o','ConnectTimeout=10',*[str(p) for p in paths],'gzs:'+RR+'/'],check=True,timeout=120)
    script=f"""from pathlib import Path
import subprocess,os,json
r=Path('{RR}')
with (r/'controller.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'diagnose_phase419.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':''}})
print(json.dumps({{'PID':p.pid,'root':str(r)}}))
"""
    data=ssh(script);(L/'LAUNCH.json').write_bytes(data);print(data.decode())
if __name__=='__main__':main()
