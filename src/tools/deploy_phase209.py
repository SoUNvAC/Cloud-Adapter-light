import pathlib,hashlib,json,zipfile,subprocess,time
r=pathlib.Path('/home/scv/shared/phase209_20261020');zpath=pathlib.Path('/home/scv/shared/phase209_20261020_bootstrap.zip')
assert hashlib.sha256(zpath.read_bytes()).hexdigest()=='9e790d141e82e84c856f70c155d45ac73d59c51d5fae795d5db7bebdec382c9a'
if (r/'LAUNCH.json').exists():
 print((r/'LAUNCH.json').read_text());raise SystemExit(0)
r.mkdir(exist_ok=False)
with zipfile.ZipFile(zpath) as z:
 for n in z.namelist():
  assert '..' not in pathlib.Path(n).parts and not n.startswith('/')
 z.extractall(r)
lock=json.loads((r/'EXECUTION_LOCK.json').read_text())
assert hashlib.sha256((r/'code/execute_phase209.py').read_bytes()).hexdigest()==lock['new_code_sha256']
parents=[pathlib.Path('/home/scv/shared')/n for n in ['phase205_20261019','phase206_20261019','phase207_20261020','phase208_20261020']]
cache=[str(q) for p in parents for q in p.glob('*Calib*')]
(r/'REMOTE_CACHE_AUDIT.json').write_text(json.dumps(dict(scanned_fixed_V4_roots=[str(p) for p in parents],calibration_candidates=cache),indent=2))
assert not cache,'inspect any existing V4 calibration cache before launching'
with (r/'console.log').open('x') as o,(r/'error.log').open('x') as e:
 p=subprocess.Popen(['/home/scv/shared/phase111_ocm_runtime/runtime_cuda128/bin/python','-B',str(r/'code/execute_phase209.py')],cwd='/home/scv/Cloud-Adapter-light',stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(PID=p.pid,epoch=time.time(),code_sha256=lock['new_code_sha256'],bootstrap_sha256=hashlib.sha256(zpath.read_bytes()).hexdigest(),local_baseline_A_passed_before_launch=True)
(r/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
