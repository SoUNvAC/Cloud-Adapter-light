from pathlib import Path
import json,os
R=Path('/home/scv/shared/phase218_20261023')
def main():
 d={}
 for n in ['PID','RUN_STATUS.json','EXIT_CODE']:
  p=R/n
  if p.exists():
   try:d[n]=json.loads(p.read_text())
   except ValueError:d[n]=p.read_text().strip()
 if 'PID' in d:
  try:os.kill(int(d['PID']),0);d['alive']=True
  except ProcessLookupError:d['alive']=False
 for n in ['FAILURE.txt','RESOURCE_FAILURE.txt']:
  if (R/n).exists():d[n]=(R/n).read_text()[-2500:]
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
