"""CPU-only analysis recovery; preserves original sealed inference script and scores."""
from pathlib import Path
import json,subprocess,base64,hashlib,time
B=Path(__file__).resolve().parents[2];O=B/'outputs/phase247/phase247_20261031';R='/home/scv/shared/phase247_20261031'
worker='''from pathlib import Path
import os,sys,json,runpy,resource,time
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1800,1800))
r=Path('/home/scv/shared/phase247_20261031');sys.path.insert(0,str(r))
from phase247_numpy_metrics import self_test
tests=self_test();source=(r/'phase247_remote.py').read_text()
assert source.count('from sklearn.metrics import roc_auc_score,average_precision_score')==1
source=source.replace('from sklearn.metrics import roc_auc_score,average_precision_score','from phase247_numpy_metrics import roc_auc_score,average_precision_score')
namespace={'__name__':'phase247_analysis_only'};exec(compile(source,'CPU_analysis_recovery','exec'),namespace)
assert (r/'INFERENCE_COMPLETE.json').exists();namespace['analyze']()
runpy.run_path(str(r/'validate_phase247.py'),run_name='__main__')
(r/'RECOVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'resolved_failure':'optional sklearn absent; equivalent NumPy binary metrics','metric_self_tests':tests,'extra_network_calls':0,'original_files_preserved':True,'epoch':time.time()},indent=2))
'''
files={'phase247_numpy_metrics.py':(B/'src/tools/phase247_numpy_metrics.py').read_bytes(),'validate_phase247.py':(B/'src/tools/validate_phase247.py').read_bytes(),'analysis_recovery.py':worker.encode()}
payload={k:base64.b64encode(v).decode() for k,v in files.items()}
code='from pathlib import Path\nimport os,json,subprocess,base64,hashlib\nr=Path('+repr(R)+')\nfiles='+repr(payload)+'\n'+'''
assert (r/'INFERENCE_COMPLETE.json').exists()
assert not (r/'ANALYSIS_RECOVERY_PID').exists(),'already launched; collect status'
for n,b64 in files.items():
 p=r/n;raw=base64.b64decode(b64)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
(r/'ANALYSIS_RECOVERY_CODE_SHA.json').write_text(json.dumps({n:hashlib.sha256(base64.b64decode(b64)).hexdigest() for n,b64 in files.items()},indent=2))
with (r/'analysis_recovery.log').open('ab') as log:p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'analysis_recovery.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(r/'ANALYSIS_RECOVERY_PID').write_text(str(p.pid));print(json.dumps({'CPU_only_recovery_PID':p.pid}))
'''
(O/'ANALYSIS_RECOVERY_CODE_SHA.json').write_text(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in files.items()},indent=2),encoding='utf-8')
for name,raw in files.items():(O/name).write_bytes(raw)
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=65)
(O/'RECOVERY_LAUNCH_RECEIPT.json').write_text(json.dumps({'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'epoch':time.time()},indent=2),encoding='utf-8');print(p.stdout);print(p.stderr)
