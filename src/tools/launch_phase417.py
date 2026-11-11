from pathlib import Path
import json,hashlib,subprocess,ast
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2];L=ROOT/'outputs/phase417/phase417_20261101';RR='/home/scv/shared/phase417_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 L.mkdir(parents=True,exist_ok=True);assert not (L/'LAUNCH.json').exists()
 paths=[ROOT/'src/tools'/n for n in ['controller_phase417.py','phase417_physics.py','phase416_optimizer.py','phase411_msre.py']]+[ROOT/'src/research_plans/phase417.md']
 for p in paths[:3]:ast.parse(p.read_text(encoding='utf-8'))
 old=ROOT/'outputs/phase400/phase400_20261031/delivery';code=json.loads((old/'CODE_LOCK.json').read_text())
 old_files={n:sha(old/n) for n in ['INPUT_LOCK.json','CODE_LOCK.json']};old_files.update(code['new_code']);old_files['PREPARED_MANIFEST.json']=json.loads((old/'INPUT_LOCK.json').read_text())['new_manifest_SHA']
 ref=ROOT/'outputs/phase411/phase411_20261101/layerwise'
 refs={'work_dirs/formal/M/'+r:sha(ref/l) for l,r in [('formal_ACTIVE_SCOPE.json','ACTIVE_SCOPE.json'),('formal_STEP_TRACE.jsonl','STEP_TRACE.jsonl')]}
 lock={'files':{p.name:sha(p) for p in paths},'old_files':old_files,'reference_files':refs,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','seed':40901,'screen_steps':1000,'max_full_promotions':0}
 (L/'LOCK.json').write_text(json.dumps(lock,indent=2));paths.append(L/'LOCK.json')
 print(ssh(f"from pathlib import Path\nr=Path('{RR}');assert not r.exists();r.mkdir();print('fresh phase417')").decode())
 subprocess.run(['scp','-o','ConnectTimeout=10',*[str(p) for p in paths],'gzs:'+RR+'/'],check=True,timeout=120)
 script=f"""from pathlib import Path
import subprocess,os,json
r=Path('{RR}')
with (r/'controller.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'controller_phase417.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}})
print(json.dumps({{'PID':p.pid,'root':str(r)}}))
"""
 data=ssh(script);(L/'LAUNCH.json').write_bytes(data);print(data.decode())
if __name__=='__main__':main()
