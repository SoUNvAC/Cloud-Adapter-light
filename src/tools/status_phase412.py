from collect_phase241 import ssh
print(ssh("""from pathlib import Path
import json
out={}
for phase in [412,413]:
 r=Path(f'/home/scv/shared/phase{phase}_20261101');v={};out[str(phase)]=v
 if phase==412 and (r/'REPORT_COMPLETE.json').exists():
  v['report_complete']=True
  continue
 for name in ['STATUS.json','PROMOTION.json','REPORT_STATUS.json','REPORT_FAILURE.json','FAILURE.json']:
  if (r/name).exists():v[name]=json.loads((r/name).read_text())
 if (r/'SCREEN.json').exists():v['screen']={m:round(x['rows'][0]['mIoU'],4) for m,x in json.loads((r/'SCREEN.json').read_text()).items()}
 if (r/'EXIT_CODE').exists():v['exit_code']=(r/'EXIT_CODE').read_text()
 if (r/'REPORT_COMPLETE.json').exists():v['report_complete']=True
 mode=v.get('STATUS.json',{}).get('mode','');p=r/mode/'RUN_STATUS.json'
 if p.exists():
  x=json.loads(p.read_text());v['step']=(x.get('current') or {}).get('step')
 if 'FAILURE.json' in v or 'REPORT_FAILURE.json' in v:
  p=r/('report.log' if 'REPORT_FAILURE.json' in v else mode+'.log')
  if p.exists():v['log_tail']=p.read_text(errors='replace')[-5000:]
print(json.dumps(out))
""").decode())
