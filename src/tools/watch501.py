"""Reconnectable read-only snapshots through training and postprocessing."""
import json,time
from collect_phase241 import ssh
script=r'''
from pathlib import Path
import json,re,os
r=Path('/home/scv/shared/phase501_20261111')
out=json.loads((r/'STATUS.json').read_text())
p=r/'texture.log'
if p.exists():
 text=p.read_text(errors='replace')
 steps=re.findall(r'Iter\(train\) \[\s*(\d+)/40000\]',text)
 out['step']=int(steps[-1]) if steps else 0
 out['last_classes']={k:float(v) for k,v in re.findall(r'\|\s*(surface_visible|cloud|shadow)\s*\|\s*([0-9.]+)',text)}
 vals=re.findall(r'Iter\(val\) \[268/268\].*?mIoU: ([0-9.]+)',text)
 out['last_mIoU']=float(vals[-1]) if vals else None
for name in ['POSTPROCESS.json','FAILURE.txt','POSTPROCESS_FAILURE.txt']:
 p=r/name
 if p.exists():out[name]=p.read_text()[-2000:]
if out['state']=='texture':out['pid_exists']=Path('/proc',str(out['pid'])).exists()
print(json.dumps(out))
'''
while True:
 try:
  out=json.loads(ssh(script))
  print(json.dumps(out),flush=True)
  if out['state']=='failed' or json.loads(out.get('POSTPROCESS.json','{}')).get('state') in ['complete','failed']:break
  if out.get('pid_exists') is False:raise RuntimeError('Training PID disappeared')
 except Exception as e:print('WATCH_ERROR '+str(e),flush=True)
 time.sleep(45)
