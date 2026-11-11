from pathlib import Path
import os,json
R=Path('/home/scv/shared/phase219_20261023')
def main():
 d={}
 for n in ['PID','CONTROLLER_STATUS.json','EXIT_CODE']:
  if (R/n).exists():
   try:d[n]=json.loads((R/n).read_text())
   except ValueError:d[n]=(R/n).read_text().strip()
 if 'PID' in d:
  try:os.kill(int(d['PID']),0);d['alive']=True
  except ProcessLookupError:d['alive']=False
 child=d.get('CONTROLLER_STATUS.json',{}).get('child_root')
 if child:
  for n in ['RUN_STATUS.json','FAILURE.txt','RESOURCE_FAILURE.txt']:
   if (R/child/n).exists():
    t=(R/child/n).read_text();d[child+'/'+n]=json.loads(t) if n.endswith('json') else t[-2200:]
 if (R/'CONTROLLER_FAILURE.txt').exists():d['controller_failure']=(R/'CONTROLLER_FAILURE.txt').read_text()[-2200:]
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
