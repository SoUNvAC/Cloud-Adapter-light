"""Reuse accepted225 fixed evaluation algebra; W1/W10 diagnostic, no replay training."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2];T=R/'src/tools'
def names(t):
 return t.replace("'R'","'W1'").replace("'Q'","'W10'").replace('R_prediction_SHA','W1_prediction_SHA').replace('Q_prediction_SHA','W10_prediction_SHA').replace('Q_and_D_semantic','W10_and_D_semantic').replace('candidate_shared_Q_head','candidate_shared_W10_head')
def put(n,t):ast.parse(t);(T/n).write_text(t,encoding='utf-8')
t=names((T/'evaluate_phase225.py').read_text()).replace('phase225','phase227').replace('phase227_20261024','phase227_20261025').replace('<140000','<125000').replace('def main():','def run_domain():').replace("assert prior['units']==64016","assert prior['units']>=32008").replace('size=sum(p.stat().st_size for p in ROOT.rglob','size=sum(p.stat().st_size for p in TOP.rglob').replace("teacher_cache={}","teacher_cache=SHARED_TEACHER").replace("from controller_phase227 import host","from controller_phase227 import host").replace("SOURCE_TO_PARENT,L8_TO_PARENT","SOURCE_TO_PARENT,L8_TO_PARENT,SPARCS_TO_PARENT").replace("==[535,963,2167]","==([535,10,10] if DOMAIN=='SPARCS' else [535,963,2167])").replace("maps={'source':SOURCE_TO_PARENT,'val':L8_TO_PARENT,'test':L8_TO_PARENT}","maps={'source':SOURCE_TO_PARENT,'val':SPARCS_TO_PARENT if DOMAIN=='SPARCS' else L8_TO_PARENT,'test':SPARCS_TO_PARENT if DOMAIN=='SPARCS' else L8_TO_PARENT}")
t=t.replace("ROOT=Path('/home/scv/shared/phase227_20261025');OUT=ROOT/'evaluation'","TOP=Path('/home/scv/shared/phase227_20261025');ROOT=TOP/'SPARCS';OUT=ROOT/'evaluation';DOMAIN='SPARCS';SHARED_TEACHER={}")
t=t.replace("'path':'same frozen features as C endpoint; shared/full semantic gate passed'","'path':'same frozen features as W1 endpoint; shared/full semantic gate passed'")
t=t.replace("save(f'{s}_Q_a1__source.json'","save(f'{s}_W10_a1__source.json'")
old="assert counts=={'Source_parity_full_segmentor':3,'candidate_full_segmentor':57570,'candidate_shared_W10_head':1070,'diagnostic_teacher_head':535,'shared_D_gate_full':2,'historical_D_gate_full':2,'historical_D_head':1070};"
new="expected={'Source_parity_full_segmentor':3,'candidate_full_segmentor':7810 if DOMAIN=='SPARCS' else 57570,'candidate_shared_W10_head':1070,'shared_D_gate_full':2,'historical_D_gate_full':2,'historical_D_head':1070};expected.update({'diagnostic_teacher_head':535} if DOMAIN=='SPARCS' else {});assert counts==expected;"
assert old in t;t=t.replace(old,new)
t=t.replace("if __name__=='__main__':main()",'''def main():
 global ROOT,OUT,DOMAIN
 prior=js(TOP/'BUDGET_STATE.json');units=prior['units'];seconds=prior['GPU_seconds'];assert units==32008
 for domain in ['SPARCS','L8']:
  DOMAIN=domain;ROOT=TOP/domain;OUT=ROOT/'evaluation';(ROOT/'BUDGET_STATE.json').write_text(json.dumps({'units':units,'GPU_seconds':seconds}));run_domain();v=js(OUT/'COMPLETE.json');assert v['total_new_units']==(10492 if domain=='SPARCS' else 59717);units+=v['total_new_units'];seconds+=v['GPU_seconds']
  import torch
  torch.cuda.empty_cache()
 assert units==102217 and len(SHARED_TEACHER)==535
 (TOP/'EVALUATION_COMPLETE.json').write_text(json.dumps({'units':units,'new_eval_units':70209,'GPU_seconds_cumulative':seconds,'teacher_prediction_inputs':535,'teacher_train_forward':0,'teacher_eval_head_forward':535,'teacher_cache_reused_across_domains':True}));(TOP/'EVALUATION_EXIT_CODE').write_text('0\\n')
if __name__=='__main__':main()''')
put('evaluate_phase227.py',t)
t=(T/'check_phase225.py').read_text().replace('phase225_20261024','phase227_20261025');t=t.replace("for n in ['RUN_STATUS.json','FAILURE.txt','FAILURE_COST.json','RESOURCE_FAILURE.txt']:","for n in ['RUN_STATUS.json','FAILURE.txt','FAILURE_COST.json','RESOURCE_FAILURE.txt','CONTROLLER_FAILURE.txt']:")
put('check_phase227.py',t)
print('227 evaluator/check generated')
