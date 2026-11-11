from pathlib import Path
import json,hashlib
from collect_phase241 import ssh
L=Path(__file__).resolve().parents[2]/'outputs/phase407/phase407_20261101';RR='/home/scv/shared/phase407_20261101'
def main():
 script=f"""from pathlib import Path
import json,os,csv
r=Path('{RR}');v={{}}
for n in ['STATUS.json','COMPLETE.json','FAILURE.json']:
 if (r/n).exists():v[n]=json.loads((r/n).read_text())
for name in ['405_40501','405_40502','406_40501','406_40502']:
 for file in ['STATUS.json','SELECTION.json','RUN_INFO.json']:
  p=r/name/file
  if p.exists():
   x=json.loads(p.read_text());v[name+'_'+file]={{k:x[k] for k in ['case','domain','index','calls','alpha','target_mIoU','absolute_error','seconds','passed'] if k in x}}
 p=r/name/'summary.csv'
 if p.exists():
  with p.open() as f:
   for x in csv.DictReader(f):
    if x['case']=='selected':v[name+'_selected_metrics']={{k:x[k] for k in ['alpha','target_mIoU','source_mIoU']}}
if (r/'EXIT_CODE').exists():v['exit_code']=(r/'EXIT_CODE').read_text()
if v.get('exit_code')=='1':
 stage=v.get('STATUS.json',{{}}).get('job');p=r/(str(stage)+'.log')
 if p.exists():v['log_tail']=p.read_text(errors='replace')[-4000:]
print(json.dumps(v))
"""
 data=ssh(script);(L/'LAST_STATUS.json').write_bytes(data);v=json.loads(data)
 print(json.dumps(v))
 if v.get('exit_code')!='0' or (L/'TRANSFER_ACCEPTED.json').exists():return
 script=f"""from pathlib import Path
import json,hashlib
r=Path('{RR}');manifest=json.loads((r/'DELIVERY_MANIFEST.json').read_text());out={{}}
for n,h in manifest.items():
 b=(r/'delivery'/n).read_bytes();assert len(b)<32*1024**2 and hashlib.sha256(b).hexdigest()==h;out[n]={{'text':b.decode(),'SHA':h}}
print(json.dumps(out))
"""
 files=json.loads(ssh(script));manifest={}
 for n,v in files.items():
  p=L/n;assert p.resolve().is_relative_to(L.resolve());p.parent.mkdir(parents=True,exist_ok=True);b=v['text'].encode();assert hashlib.sha256(b).hexdigest()==v['SHA'];p.write_bytes(b);manifest[n]=v['SHA']
 (L/'TRANSFER_ACCEPTED.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
