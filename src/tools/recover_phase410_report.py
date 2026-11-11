"""Resume CPU-only reporting after completed training; preserve failure evidence."""
from pathlib import Path
import json,time,textwrap,hashlib,shutil
import controller_phase410 as c
R=c.R

def main():
 assert (R/'EXIT_CODE').read_text().strip()=='1'
 failure=c.js(R/'FAILURE.json')
 assert "serial/work_dirs/formal/M/STEP_TRACE.jsonl" in failure['traceback'] and 'FileNotFoundError' in failure['traceback']
 for n,h in c.js(R/'LOCK.json')['new_files'].items():assert c.sha(R/n)==h
 calls=0
 for mode in ['baseline','adaptive']:
  assert c.js(R/mode/'COMPLETE.json')['passed']
  ledger=[json.loads(x) for x in (R/mode/'CALL_LEDGER.jsonl').read_text().splitlines()]
  assert len(ledger)==4943
  assert sum(x['kind']=='train_full_input' for x in ledger)==4120
  calls+=len(ledger)
 assert calls==9886
 shutil.copy2(R/'EXIT_CODE',R/'FAILED_EXIT_CODE')
 script=(R/'controller_phase410.py').read_text()
 tail=script[script.index(' assert totalcalls==9886'):script.index("if __name__=='__main__':")]
 assert tail.count("serial/work_dirs")==1 and tail.count("parallel/work_dirs")==1
 tail=tail.replace('serial/work_dirs','baseline/work_dirs').replace('parallel/work_dirs','adaptive/work_dirs')
 elapsed=time.time()-(R/'LAUNCH.json').stat().st_mtime
 c.START=time.monotonic()-elapsed
 c.__dict__.update(totalcalls=9886,totalupdates=8240)
 exec(compile(textwrap.dedent(tail),'phase410_reporting_recovery','exec'),c.__dict__)
 after=sum(len((R/m/'CALL_LEDGER.jsonl').read_text().splitlines()) for m in ['baseline','adaptive'])
 assert after==calls
 c.save('REPORT_RECOVERY.json',{'passed':True,'reason':'post-training directory names serial/parallel corrected to baseline/adaptive','additional_training_updates':0,'additional_model_forwards':0,'calls_before':calls,'calls_after':after,'original_failure_preserved':'FAILURE.json','recovery_code_SHA':c.sha(Path(__file__))})
 for n in ['FAILURE.json','FAILED_EXIT_CODE','REPORT_RECOVERY.json']:shutil.copy2(R/n,R/'delivery'/n)
 c.save('DELIVERY_MANIFEST.json',{p.relative_to(R/'delivery').as_posix():c.sha(p) for p in (R/'delivery').rglob('*') if p.is_file()})
 print('CPU-only report recovery complete')
if __name__=='__main__':main()
