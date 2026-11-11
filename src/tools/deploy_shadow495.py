from pathlib import Path
import json,hashlib,subprocess
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'outputs/phase495/phase495_20261109'
R.mkdir(parents=True,exist_ok=True)
files=list((ROOT/'src/tools').glob('*495.py'))+list((ROOT/'src/tools').glob('*495.sh'))+[ROOT/'src/research_plans/phase495.md']
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/'INITIAL_CODE_SHA.json').write_text(json.dumps(manifest,indent=2))
print(ssh("""from pathlib import Path
import shutil,subprocess,torch
p=Path('/mnt/d/CloudAdapterExperiments/phase495_20261109');q=Path('/home/scv/shared/phase495_20261109')
assert not p.exists() and not q.exists()
free=shutil.disk_usage(p.parent).free
assert free>12*1024**3,free
assert torch.cuda.is_available()
print('gpu',torch.cuda.get_device_name(0),'free_total_bytes',torch.cuda.mem_get_info())
p.mkdir();q.symlink_to(p,target_is_directory=True)
print('free_bytes',free)
""").decode(),flush=True)
subprocess.run(['scp','-q',*[str(p) for p in files],str(R/'INITIAL_CODE_SHA.json'),'gzs:/home/scv/shared/phase495_20261109/'],check=True)
print(ssh("""from pathlib import Path
import hashlib,json,subprocess,sys
r=Path('/home/scv/shared/phase495_20261109')
for n,h in json.loads((r/'INITIAL_CODE_SHA.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,str(r/'audit_shadow495.py')],check=True,cwd=r)
print('DEPLOY_AND_UNIT_ACCEPTED')
""").decode(),flush=True)
