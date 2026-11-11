from pathlib import Path
import json,os
R=Path('/home/scv/shared/phase220_20261024')
def main():
 d={}
 for n in ['PID','RUN_STATUS.json','EXIT_CODE','FAILURE.txt']:
  if (R/n).exists():
   t=(R/n).read_text()
   try:d[n]=json.loads(t)
   except ValueError:d[n]=t[-2000:]
 if 'PID' in d:
  try:os.kill(int(d['PID']),0);d['alive']=True
  except ProcessLookupError:d['alive']=False
 for mode in ['val','replay']:
  for n in ['RUN_STATUS.json','FAILURE.txt','EXIT_CODE']:
   if (R/mode/n).exists():
    t=(R/mode/n).read_text()
    try:d[mode+'/'+n]=json.loads(t)
    except ValueError:d[mode+'/'+n]=t[-2000:]
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
