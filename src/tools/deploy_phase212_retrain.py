"""Idempotent launch of the SHA-locked, user-authorized two-head recovery."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import zipfile

ROOT=Path('/home/scv/shared/phase212_retrain_20261020')
ARCHIVE=Path('/home/scv/shared/phase212_retrain_20261020_bootstrap.zip')
EXPECTED='20bfbdce13ade2a704a3dd8b4448f3bd5f439d468a47003238799fba62075286'
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==EXPECTED
if (ROOT/'LAUNCH.json').exists():
    print((ROOT/'LAUNCH.json').read_text());raise SystemExit(0)
ROOT.mkdir(exist_ok=False)
with zipfile.ZipFile(ARCHIVE) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(ROOT)
lock=json.loads((ROOT/'EXECUTION_LOCK.json').read_text())
for name,h in lock['code_files'].items():assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==h
assert hashlib.sha256((ROOT/'PREFLIGHT_ACCEPTED.json').read_bytes()).hexdigest()==lock['preflight_sha256']
with (ROOT/'controller.console.log').open('x') as o,(ROOT/'controller.error.log').open('x') as e:
    p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B',str(ROOT/'code/controller_phase212_retrain.py')],cwd='/home/scv/Cloud-Adapter-light/src',stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
receipt={'PID':p.pid,'epoch':time.time(),'stage':'two_head_retraining_dispatched','bootstrap_sha256':EXPECTED,'training_calls_cap':16000,'new_persistent_cap_bytes':500000000,'no_timer_created':True}
(ROOT/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
