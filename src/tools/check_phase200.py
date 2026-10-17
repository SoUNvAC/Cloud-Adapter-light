"""One compact status snapshot; never launches, waits, or reads predictions."""
import os,json
from phase200_common import *
def main():
    result={}
    for stage in ['acquire','infer']:
        d=ROOT/stage
        if not d.exists():result[stage]={'status':'not_started'};continue
        p=d/'PID';pid=int(p.read_text()) if p.exists() else None
        alive=bool(pid and Path(f'/proc/{pid}').exists() and Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z')
        s=load(d/'STATUS.json') if (d/'STATUS.json').exists() else {}
        result[stage]={k:s.get(k) for k in ['status','current','received_bytes','verified_objects','model_calls','main_calls','construction_calls','seconds','gpu_phase_seconds','reason'] if k in s}
        result[stage].update(PID=pid,alive=alive,EXIT_CODE=(d/'EXIT_CODE').read_text().strip() if (d/'EXIT_CODE').exists() else None,
            completed_versions=[k for k,v in s.get('versions',{}).items() if v['status']=='completed'])
        if s.get('status')=='failed':result[stage]['error_tail']=s.get('error','')[-1400:]
    result['points_sealed']=(ROOT/'POINTS_SEALED.json').exists();result['result_exists']=(ROOT/'delivery/RESULT.md').exists()
    result['partial_bytes']=sum(p.stat().st_size for p in (ROOT/'assets').glob('*.partial')) if (ROOT/'assets').exists() else 0
    print(json.dumps(result))
if __name__=='__main__':main()
