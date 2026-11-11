"""Idempotent guarded launch; invoked over authorized existing SSH."""
import pathlib,hashlib,json,zipfile,subprocess,time
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
r=pathlib.Path('/home/scv/shared/phase210_20261020');zpath=pathlib.Path('/home/scv/shared/phase210_20261020_bootstrap.zip');assert sha(zpath)=='4e8d87665f42cb08d5f03d6e45607804bcf5d0ce547acdb65123fdf0cffff400'
if (r/'LAUNCH.json').exists():
 print((r/'LAUNCH.json').read_text());raise SystemExit(0)
r.mkdir(exist_ok=False)
with zipfile.ZipFile(zpath) as z:
 for n in z.namelist():assert '..' not in pathlib.Path(n).parts and not n.startswith('/')
 z.extractall(r)
lock=json.loads((r/'EXECUTION_LOCK.json').read_text());assert sha(r/'code/execute_phase210.py')==lock['code_sha256'];assert sha(r/'code/execute_phase209.py')==lock['common_helper_sha256']
other=pathlib.Path('/home/scv/Cloud-Adapter-light/src/tools/execute_phase209.py');assert not other.exists() or sha(other)==lock['common_helper_sha256']
plan=json.loads((r/'MISSING_PLAN.json').read_text());verified=[]
for q in plan['rows']:
 p=pathlib.Path(q['cache_remote_path']);assert sha(p)==q['cache_sha256'];verified.append(q['product_id'])
candidates=[]
for dirname in ['phase205_20261019','phase208_20261020']:
 for p in (pathlib.Path('/home/scv/shared')/dirname).rglob('*'):
  if p.is_file() and p.suffix in ['.npz','.npy','.pt'] and any(s in p.name.lower() for s in ['raw_logits','conditional','nonshadow']):candidates.append(str(p))
assert not candidates,'inspect raw/conditional cache candidates before any new forward'
preflight=dict(verified_original_score_caches=len(verified),raw_or_conditional_candidates=candidates,planned_main_calls=plan['main_calls'],bootstrap_sha256=sha(zpath));(r/'REMOTE_PREFLIGHT.json').write_text(json.dumps(preflight,indent=2))
with (r/'console.log').open('x') as o,(r/'error.log').open('x') as e:
 p=subprocess.Popen(['/home/scv/shared/phase111_ocm_runtime/runtime_cuda128/bin/python','-B',str(r/'code/execute_phase210.py')],cwd='/home/scv/Cloud-Adapter-light',stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(PID=p.pid,epoch=time.time(),code_sha256=lock['code_sha256'],preflight=preflight)
(r/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
