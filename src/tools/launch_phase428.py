"""Upload an immutable run bundle and detach exactly one source retraining job."""
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path
from collect_phase241 import ssh

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / 'outputs/phase428/phase428_20261101'
REMOTE = '/home/scv/shared/phase428_20261101'


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    assert not (LOCAL / 'LAUNCH.json').exists()
    files = {}
    for name in ['phase428_model.py', 'phase427_fused.py', 'phase411_msre.py', 'worker_phase428.py']:
        content = (REPO / 'src/tools' / name).read_text(encoding='utf-8')
        ast.parse(content)
        files[name] = content.encode()
    config = (REPO / 'src/configs/protocol/phase428_fused_source_40k.py').read_text()
    config = config.replace('./phase64_source_parent_rgb.py',
                            '/home/scv/Cloud-Adapter-light/src/configs/protocol/phase64_source_parent_rgb.py')
    ast.parse(config)
    files['phase428_fused_source_40k.py'] = config.encode()
    files['phase428.md'] = (REPO / 'src/research_plans/phase428.md').read_bytes()
    lock = {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}
    files['CODE_LOCK.json'] = json.dumps(lock, indent=2).encode()
    for name, content in files.items():
        (LOCAL / name).write_bytes(content)
    payload = {n: base64.b64encode(v).decode() for n, v in files.items()}
    script = """from pathlib import Path
import os,json,base64,hashlib,subprocess,shutil
r=Path(REMOTE)
r.mkdir(exist_ok=True)
assert not (r/'LAUNCH.json').exists() and not (r/'console.log').exists(), 'existing run; inspect before relaunch'
assert shutil.disk_usage(r).free > 15*1024**3, 'insufficient disk space'
smi='/usr/lib/wsl/lib/nvidia-smi'
assert not subprocess.check_output([smi,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU busy'
for n,v in PAYLOAD.items():
 (r/n).write_bytes(base64.b64decode(v))
for n,h in json.loads((r/'CODE_LOCK.json').read_text()).items():
 assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONPATH':str(r)+':/home/scv/Cloud-Adapter-light/src'}
with (r/'console.log').open('x') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'worker_phase428.py')],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'PID':p.pid,'root':str(r),'max_iters':40000,'mode':'integrated_source_retrain'}
(r/'LAUNCH.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
""".replace('REMOTE', repr(REMOTE)).replace('PAYLOAD', repr(payload))
    packed = base64.b64encode(zlib.compress(script.encode())).decode()
    data = ssh("import zlib,base64;exec(zlib.decompress(base64.b64decode(" + repr(packed) + ")))")
    (LOCAL / 'LAUNCH.json').write_bytes(data)
    print(data.decode())


if __name__ == '__main__':
    main()
