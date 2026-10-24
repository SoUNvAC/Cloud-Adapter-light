"""Add missing GPU memory telemetry before any Phase303 forward is permitted."""
import shutil
from phase303_common import R,ROOT,load,write,sha

assert load(ROOT/'STATUS.json')['status']=='waiting_for_gpu'
assert set(load(ROOT/'STAGES.json'))=={'cached'}
assert not (ROOT/'CALLS.jsonl').exists()
lock=load(ROOT/'EXECUTION_LOCK.json')
name='src/tools/worker_phase303.py'
assert sha(ROOT/'code/worker_phase303.py')==lock['code'][name]
for path,digest in lock['code'].items():
    if path!=name:assert sha(R/path)==digest,path
assert not (ROOT/'INITIAL_EXECUTION_LOCK.json').exists()
shutil.copyfile(ROOT/'EXECUTION_LOCK.json',ROOT/'INITIAL_EXECUTION_LOCK.json')
shutil.copyfile(R/name,ROOT/'code/worker_phase303_with_gpu_telemetry.py')
lock['code'][name]=sha(R/name)
lock['amendment']='Added peak GPU memory/runtime receipt only, after cached CPU stage and before first forward; original code and lock retained'
write(ROOT/'EXECUTION_LOCK.json',lock)
print('Pre-forward telemetry amendment recorded; no model call')
