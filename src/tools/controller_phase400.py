"""One sequential GPU queue, no early-loss pruning and no automatic Phase401."""
from phase400_common import *
import sys,traceback,signal
def run(stage,args,gpu=False):
 save('CONTROLLER_STATUS.json',{'PID':os.getpid(),'stage':stage,'GPU_admission':gpu,'epoch':time.time()});log=R/'logs'/(stage+'.log');log.parent.mkdir(exist_ok=True);assert not log.exists()
 with log.open('x') as f:
  started=time.monotonic();p=subprocess.Popen([PY,'-B','-u',*args],cwd=R/'code',stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(R/'code')+':'+str(SRC)+':'+str(SRC/'tools'),'XFORMERS_DISABLED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TMPDIR':str(R/'scratch')});save('WORKER.json',{'PID':p.pid,'stage':stage});limit=max(1,43200-prior_gpu_seconds()) if gpu else 14400
  try:rc=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   save('CONTROLLER_TIMEOUT.json',{'stage':stage,'seconds':time.monotonic()-started,'GPU_stage':gpu});raise
 with (R/'STAGE_JOURNAL.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'returncode':rc,'seconds':time.monotonic()-started,'GPU_stage':gpu})+'\n')
 if rc:raise RuntimeError(stage+' exited '+str(rc)+'; confirm exited; preserve records; at most two directed repairs within remaining GPU budget')
def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();(R/'scratch').mkdir(exist_ok=True);save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'formal_updates_planned':12000,'GPU_normal_plan_cap_seconds':36000,'GPU_all_attempts_cap_seconds':43200,'max_directed_repairs':2})
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h,n
 run('admission',[str(R/'code/admit_phase400.py')]);run('gate',[str(R/'code/gate_phase400.py')],True)
 for arm in ('MH','M','H'):run('short_'+arm,[str(R/'code/worker_phase400.py'),'short',arm],True)
 reports={arm:js(R/('short_'+arm+'_COMPLETE.json')) for arm in ('MH','M','H')};gate=js(R/'CORRECTNESS_GATE.json');step=sum(4000*(v['mean_step_seconds']+v['host_gap_mean']) for v in reports.values());ev=max(v['one_development_model_seconds']+v['one_input_load_seconds'] for v in gate['evaluation_probes'])*1.5;evals=(535+32)+(3*8*32)+(3*2*2*535);construction=sum(v['build_and_gate_seconds'] for v in gate['reports'])*6;estimated=prior_gpu_seconds()+step+evals*ev+construction+300
 save('SHORT_RUN.json',{'passed':estimated<=36000,'reports':reports,'formal_training_estimated_seconds':step,'full_evaluation_upper_count':evals,'evaluation_seconds_per_input_conservative':ev,'build_and_engineering_upper_seconds':construction+300,'spent_GPU_stage_seconds':prior_gpu_seconds(),'normal_plan_estimated_seconds':estimated,'normal_cap':36000,'all_attempts_cap':43200,'rule':'20 warm +100 timed each; no loss-based arm pruning; full scheduled development plus all best/final branch-on/off source'})
 assert estimated<=36000,'Measured plan exceeds normal 10h; revise before formal training, no silent omission'
 save('FORMAL_ADMISSION.json',{'passed':True,'epoch':time.time(),'short_run_SHA':sha(R/'SHORT_RUN.json'),'gate_SHA':sha(R/'CORRECTNESS_GATE.json'),'input_lock_SHA':sha(R/'INPUT_LOCK.json'),'code_SHA':{p.name:sha(p) for p in (R/'code').glob('*') if p.is_file()}})
 run('Source_baseline',[str(R/'code/worker_phase400.py'),'evaluate','S',str(SOURCE),'on','Source'],True)
 for arm in ('MH','M','H'):run('formal_'+arm,[str(R/'code/worker_phase400.py'),'formal',arm],True)
 # Selection uses only the scheduled development pooled mIoU, earliest step on an exact tie.
 endpoints=[]
 for arm in ('MH','M','H'):
  values=[{'step':s,'path':R/'evaluations'/arm/('step_'+str(s))/'METRICS.json','mIoU':js(R/'evaluations'/arm/('step_'+str(s))/'METRICS.json')['mIoU']} for s in range(500,4001,500)];best=max(values,key=lambda v:(v['mIoU'],-v['step']));folder=R/'work_dirs/formal'/arm;bp=folder/('best_mIoU_iter_'+str(best['step'])+'.pth');fp=folder/'iter_4000.pth';assert bp.exists() and fp.exists();endpoints.extend([{'arm':arm,'role':'best','step':best['step'],'mIoU':best['mIoU'],'checkpoint':str(bp),'checkpoint_SHA':sha(bp)},{'arm':arm,'role':'final','step':4000,'mIoU':values[-1]['mIoU'],'checkpoint':str(fp),'checkpoint_SHA':sha(fp)}])
 save('SELECTION_LOCK.json',{'endpoints':endpoints,'rule':'max development pooled 3-class mIoU, earliest step tie; no Source candidate; final4000 always reported','epoch':time.time()});seen={};aliases=[]
 # CPU tensor identity deduplication: best=final weights are evaluated only once in each branch state.
 for e in endpoints:
  code="import sys;sys.path.insert(0,"+repr(str(R/'code'))+");from phase400_common import *;import torch;p=Path("+repr(e['checkpoint'])+");d=torch.load(p,map_location='cpu');print(state_sha(d['state_dict']))"
  digest=subprocess.check_output([PY,'-B','-c',code],text=True).strip();key=(e['arm'],digest)
  if key in seen:aliases.append({**e,'weight_state_SHA':digest,'alias_of':seen[key]});continue
  tag=e['arm']+'_'+e['role'];seen[key]=tag
  for branch in ('on','off'):run(tag+'_'+branch,[str(R/'code/worker_phase400.py'),'evaluate',e['arm'],e['checkpoint'],branch,tag+'_'+branch],True)
  aliases.append({**e,'weight_state_SHA':digest,'alias_of':tag})
 save('SOURCE_EVALUATION_ALIASES.json',aliases);run('cpu_report',[str(R/'code/report_phase400.py')]);save('CONTROLLER_STATUS.json',{'PID':os.getpid(),'stage':'complete','epoch':time.time()});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  save('CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc(),'cumulative_GPU_stage_seconds':prior_gpu_seconds()});(R/'EXIT_CODE').write_text('2\n');raise
