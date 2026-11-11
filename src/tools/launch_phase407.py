from pathlib import Path
import json,hashlib,ast,subprocess
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2];L=ROOT/'outputs/phase407/phase407_20261101';RR='/home/scv/shared/phase407_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 L.mkdir(parents=True,exist_ok=True);assert not (L/'LAUNCH.json').exists()
 old=ROOT/'outputs/phase400/phase400_20261031/delivery';code=json.loads((old/'CODE_LOCK.json').read_text())
 old_files={n:sha(old/n) for n in ['INPUT_LOCK.json','CODE_LOCK.json','SOURCE_EVALUATION_ALIASES.json']};old_files.update(code['new_code']);old_files['PREPARED_MANIFEST.json']=json.loads((old/'INPUT_LOCK.json').read_text())['new_manifest_SHA']
 paths=[ROOT/'src/tools/controller_phase407.py',ROOT/'src/tools/evaluate_phase407.py',ROOT/'src/research_plans/phase407.md']
 for p in paths[:2]:ast.parse(p.read_text(encoding='utf-8-sig'))
 jobs=[]
 for phase in [405,406]:
  for seed in [40501,40502]:
   previous=ROOT/f'outputs/phase{phase}/phase{phase}_20261101'
   manifest=json.loads((previous/'LOCAL_MANIFEST.json').read_text())
   for file in [f'{seed}/summary.csv',f'{seed}/per_group.csv',f'{seed}/COMPLETE.json']:
    assert sha(previous/file)==manifest[file]
   import csv
   with (previous/str(seed)/'summary.csv').open() as f:values=list(csv.DictReader(f))
   reference_root=ROOT/'outputs/phase405/phase405_20261101'
   with (reference_root/str(seed)/'summary.csv').open() as f:reference=float(next(x for x in csv.DictReader(f) if x['case']=='weak_5')['target_mIoU'])
   points=sorted((float(x['case']),float(x['target_mIoU'])) for x in values if x['case'] in ['0.4','0.45','0.5','1.0'])
   brackets=[(b[0]-a[0],a[0],b[0]) for a,b in zip(points,points[1:]) if a[1]<reference<b[1]]
   assert brackets
   _,lo,hi=min(brackets)
   info=json.loads((previous/str(seed)/'COMPLETE.json').read_text())
   jobs.append({'name':f'{phase}_{seed}','phase':phase,'seed':seed,'prior_root':f'/home/scv/shared/phase{phase}_20261101/{seed}','checkpoint_SHA':info['checkpoint_SHA'],'groups_SHA':sha(previous/str(seed)/'per_group.csv'),'summary_SHA':sha(previous/str(seed)/'summary.csv'),'target_reference':reference,'bracket':[lo,hi]})
 lock={'jobs':jobs,'old_files' :old_files,'new_files':{p.name:sha(p) for p in paths},'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','seeds':[40501,40502],'max_wall_seconds':3600}
 (L/'LOCK.json').write_text(json.dumps(lock,indent=2));paths.append(L/'LOCK.json')
 print(ssh(f"from pathlib import Path\nr=Path('{RR}');assert not r.exists();r.mkdir();print('fresh Phase407')").decode())
 subprocess.run(['scp','-o','ConnectTimeout=10',*[str(p) for p in paths],'gzs:'+RR+'/'],check=True,timeout=120)
 script=f"""from pathlib import Path
import subprocess,json,os,hashlib
r=Path('{RR}');lock=json.loads((r/'LOCK.json').read_text())
for n,h in lock['new_files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
with (r/'controller.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'controller_phase407.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}})
v={{'PID':p.pid,'root':str(r)}};(r/'LAUNCH.json').write_text(json.dumps(v));print(json.dumps(v))
"""
 result=ssh(script);(L/'LAUNCH.json').write_bytes(result);print(result.decode())
if __name__=='__main__':main()
