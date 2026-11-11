"""Launch once, preserving Phase400 and receiving only small reports later."""
import ast, json, hashlib, subprocess
from pathlib import Path
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2]
L=ROOT/'outputs/phase401/phase401_20261031'
R='/home/scv/shared/phase401_20261031'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    L.mkdir(parents=True,exist_ok=True)
    assert not (L/'LAUNCH.json').exists()
    worker=ROOT/'src/tools/run_phase401.py';ast.parse(worker.read_text())
    old=ROOT/'outputs/phase400/phase400_20261031/delivery'
    names=['INPUT_LOCK.json','CODE_LOCK.json','SOURCE_EVALUATION_ALIASES.json']
    lock={'old_files':{n:digest(old/n) for n in names},'H_SHA':'a86d96f3d84b5301ff7d8c8693ec1195360afbe9c9fbe4783392e669e09b6d9e','Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','worker_SHA':digest(worker)}
    old_code=json.loads((old/'CODE_LOCK.json').read_text())
    lock['old_files'].update(old_code['new_code'])
    lock['old_files']['PREPARED_MANIFEST.json']=json.loads((old/'INPUT_LOCK.json').read_text())['new_manifest_SHA']
    (L/'INPUT_EXPECTED.json').write_text(json.dumps(lock,indent=2))
    print(ssh(f"from pathlib import Path\np=Path('{R}');assert not p.exists();p.mkdir();print('created fresh Phase401')").decode())
    paths=[worker,L/'INPUT_EXPECTED.json',ROOT/'src/research_plans/phase401.md']
    subprocess.run(['scp','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',*[str(p) for p in paths],'gzs:'+R+'/'],check=True,timeout=120)
    script=f"""from pathlib import Path
import json,hashlib,subprocess,os
r=Path('{R}');assert not (r/'LAUNCH.json').exists()
assert hashlib.sha256((r/'run_phase401.py').read_bytes()).hexdigest()=='{digest(worker)}'
with (r/'run.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'run_phase401.py')],stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True,env={{**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}})
v={{'PID':p.pid,'root':str(r)}};(r/'LAUNCH.json').write_text(json.dumps(v));print(json.dumps(v))
"""
    result=ssh(script);(L/'LAUNCH.json').write_bytes(result);print(result.decode())
if __name__=='__main__':main()
