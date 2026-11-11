from pathlib import Path
import json,hashlib,ast,subprocess
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2];L=ROOT/'outputs/phase411/phase411_20261101';RR='/home/scv/shared/phase411_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 L.mkdir(parents=True,exist_ok=True);assert not (L/'LAUNCH.json').exists()
 old=ROOT/'outputs/phase400/phase400_20261031/delivery';code=json.loads((old/'CODE_LOCK.json').read_text())
 old_files={n:sha(old/n) for n in ['INPUT_LOCK.json','CODE_LOCK.json','SOURCE_EVALUATION_ALIASES.json']};old_files.update(code['new_code']);old_files['PREPARED_MANIFEST.json']=json.loads((old/'INPUT_LOCK.json').read_text())['new_manifest_SHA']
 paths=[ROOT/'src/tools/controller_phase411.py',ROOT/'src/tools/report_phase411.py',ROOT/'src/research_plans/phase411.md',ROOT/'src/tools/phase411_msre.py']
 reference=ROOT/'outputs/phase410/phase410_20261101'
 reference_files={n:sha(reference/n) for n in ['TARGET_SUMMARY.csv','TARGET_GROUPS.csv']}
 for local,remote in [('formal_ACTIVE_SCOPE.json','work_dirs/formal/M/ACTIVE_SCOPE.json'),('formal_STEP_TRACE.jsonl','work_dirs/formal/M/STEP_TRACE.jsonl')]:reference_files['baseline/'+remote]=sha(reference/'baseline'/local)
 reference_files.update({'baseline/code/'+n:h for n,h in json.loads((reference/'baseline/CODE_LOCK.json').read_text()).items()})
 for p in paths[:2]:ast.parse(p.read_text(encoding='utf-8-sig'))
 lock={'old_files':old_files,'new_files':{p.name:sha(p) for p in paths},'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','seed':40901,'modes':['layerwise'],'reference_files':reference_files,'reused_baseline':'phase410/baseline','max_wall_seconds':7200}
 (L/'LOCK.json').write_text(json.dumps(lock,indent=2));paths.append(L/'LOCK.json')
 print(ssh(f"from pathlib import Path\nr=Path('{RR}');assert not r.exists();r.mkdir();print('fresh Phase411')").decode())
 subprocess.run(['scp','-o','ConnectTimeout=10',*[str(p) for p in paths],'gzs:'+RR+'/'],check=True,timeout=120)
 script=f"""from pathlib import Path
import subprocess,json,os,hashlib
r=Path('{RR}');lock=json.loads((r/'LOCK.json').read_text())
for n,h in lock['new_files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
with (r/'controller.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'controller_phase411.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}})
v={{'PID':p.pid,'root':str(r)}};(r/'LAUNCH.json').write_text(json.dumps(v));print(json.dumps(v))
"""
 result=ssh(script);(L/'LAUNCH.json').write_bytes(result);print(result.decode())
if __name__=='__main__':main()
