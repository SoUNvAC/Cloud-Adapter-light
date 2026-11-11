from pathlib import Path
import hashlib,json,subprocess,time,zipfile,shutil
PKG=Path('/home/scv/shared/phase213_test_package_run02.zip');ROOT=Path('/home/scv/shared/phase213_20261021_run02');EXPECTED='pending'
if (ROOT/'LAUNCH.json').exists():
    print((ROOT/'LAUNCH.json').read_text());raise SystemExit(0)
ROOT.mkdir(exist_ok=False);code=ROOT/'code';code.mkdir()
with zipfile.ZipFile(PKG) as z:
    assert z.testzip() is None and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(code)
shutil.copyfile(code/'ALPHA_SELECTION_LOCK.json',ROOT/'ALPHA_SELECTION_LOCK.json')
assert hashlib.sha256((code/'evaluate_phase213_target_test.py').read_bytes()).hexdigest()=='b1c9942db7da30a860ce168efd1b5dfab221b4df7b86027c2d6006dca8b29fd7'
assert hashlib.sha256((code/'ALPHA_SELECTION_LOCK.json').read_bytes()).hexdigest()=='53dc42499c8c2f1af238804cf0e25b8328598f64146f81d2c0a536de544516c9'
with (ROOT/'console.log').open('x') as o,(ROOT/'error.log').open('x') as e:
    p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B',str(code/'evaluate_phase213_target_test.py')],cwd='/home/scv/Cloud-Adapter-light/src',stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
receipt={'PID':p.pid,'epoch':time.time(),'stage':'target_test_evaluation_dispatched','expected_calls':10885,'test_labels_before_lock':False,'alpha_selection_lock_sha256':hashlib.sha256((code/'ALPHA_SELECTION_LOCK.json').read_bytes()).hexdigest()}
(ROOT/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
