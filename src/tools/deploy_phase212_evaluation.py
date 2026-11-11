"""Launch the one fixed, no-training Phase212 evaluation after both fits passed."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import zipfile

ROOT=Path('/home/scv/shared/phase212_retrain_20261020_run02')
PACKAGE=Path('/home/scv/shared/phase212_retrained_evaluation_20261020.zip')
assert hashlib.sha256(PACKAGE.read_bytes()).hexdigest()=='8e147e1c6fa331a39f06a40e4bf032b81c339b25d501a63a1b2beb20f1ca3d1c'
assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
if (ROOT/'EVALUATION_LAUNCH.json').is_file():
    print((ROOT/'EVALUATION_LAUNCH.json').read_text());raise SystemExit(0)
code=ROOT/'eval_code';code.mkdir(exist_ok=False)
with zipfile.ZipFile(PACKAGE) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(code)
lock=json.loads((code/'EVALUATION_LOCK.json').read_text())
assert hashlib.sha256((code/'evaluate_phase212_retrained.py').read_bytes()).hexdigest()==lock['evaluation_code_sha256']
assert hashlib.sha256((code/'SOURCE_ROI_BINDINGS.csv').read_bytes()).hexdigest()==lock['ROI_binding_sha256']
with (ROOT/'evaluation.console.log').open('x') as o,(ROOT/'evaluation.error.log').open('x') as e:
    p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B',str(code/'evaluate_phase212_retrained.py')],cwd='/home/scv/Cloud-Adapter-light/src',stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
receipt={'PID':p.pid,'epoch':time.time(),'stage':'fixed_FP32_interpolation_evaluation_dispatched','new_training':False,'main_expected_calls':9680,'cap':16000}
(ROOT/'EVALUATION_LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
