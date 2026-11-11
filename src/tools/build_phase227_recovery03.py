"""Resume only missing target evaluations, preserve all source predictions and costs."""
from pathlib import Path
import ast,json,hashlib,zipfile
T=Path(__file__).resolve().parent;O=T.parents[1]/'outputs/phase227/phase227_20261025'
def put(n,t):ast.parse(t);(T/n).write_text(t,encoding='utf-8');(O/'code'/n).write_bytes((T/n).read_bytes())
t=(T/'evaluate_phase227.py').read_text().replace("save(f'{s}_Q_a1__source.json'","save(f'{s}_W10_a1__source.json'")
t=t.replace('OUT.mkdir(exist_ok=False)',"OUT.mkdir(exist_ok=DOMAIN=='SPARCS')").replace("ledger=(OUT/'CALL_LEDGER.jsonl').open('x',buffering=1)","ledger=(OUT/('CALL_LEDGER_RECOVERY03.jsonl' if DOMAIN=='SPARCS' else 'CALL_LEDGER.jsonl')).open('x',buffering=1)")
t=t.replace("(ROOT/'interpolated_heads').mkdir(exist_ok=False)","(ROOT/'interpolated_heads').mkdir(exist_ok=DOMAIN=='SPARCS')")
old="p=ROOT/f'interpolated_heads/{s}_{a}_a{f}.pth';torch.save({'state_dict':h,'seed':s,'arm':a,'alpha':f,'Source_SHA':SH,'formula':'CPU float64 then original dtype'},p);heads_receipt[p.name]=sha(p)"
new="""p=ROOT/f'interpolated_heads/{s}_{a}_a{f}.pth'
      if p.exists():
       stored=torch.load(p,map_location='cpu')['state_dict'];assert set(stored)==set(h) and all(torch.equal(h[n],stored[n]) for n in h)
      else:torch.save({'state_dict':h,'seed':s,'arm':a,'alpha':f,'Source_SHA':SH,'formula':'CPU float64 then original dtype'},p)
      heads_receipt[p.name]=sha(p)""";assert old in t;t=t.replace(old,new)
t=t.replace("save('INTERPOLATED_HEAD_HASHES.json',heads_receipt)","assert heads_receipt==js(OUT/'INTERPOLATED_HEAD_HASHES.json') if DOMAIN=='SPARCS' else True\n  if DOMAIN!='SPARCS':save('INTERPOLATED_HEAD_HASHES.json',heads_receipt)")
t=t.replace("joint_stream=(OUT/'JOINT_COUNTS.jsonl').open('x',buffering=1)","joint_stream=(open(os.devnull,'w') if DOMAIN=='SPARCS' else (OUT/'JOINT_COUNTS.jsonl').open('x',buffering=1))")
t=t.replace("reproduce('source');save('Source__source.json',caches['source'])", "if DOMAIN=='SPARCS':parity=js(OUT/'SOURCE_PARITY.json')\n  else:reproduce('source');save('Source__source.json',caches['source'])")
a=t.index("  for s in SEEDS:\n   for a in ['W1','W10']:\n    for f in AS[1:]:\n     if a=='W10'");b=t.index('  def score(c):',a);block=t[a:b];t=t[:a]+"  if DOMAIN!='SPARCS':\n"+''.join(' '+line if line.strip() else line for line in block.splitlines(keepends=True))+t[b:]
t=t.replace("expected.update({'diagnostic_teacher_head':535} if DOMAIN=='SPARCS' else {});assert counts==expected;", "expected.update({'diagnostic_teacher_head':535});expected={'Source_parity_full_segmentor':2,'candidate_full_segmentor':320} if DOMAIN=='SPARCS' else expected;assert counts==expected;")
a=t.index('def main():');t=t[:a]+'''def main():
 global ROOT,OUT,DOMAIN
 prior=js(TOP/'RESUME03_BUDGET.json');units=prior['units'];seconds=prior['GPU_seconds'];assert units==42178
 for domain in ['SPARCS','L8']:
  DOMAIN=domain;ROOT=TOP/domain;OUT=ROOT/'evaluation';(ROOT/'BUDGET_STATE.json').write_text(json.dumps({'units':units,'GPU_seconds':seconds}));st=js(TOP/'CONTROLLER_STATUS.json');st['child_root']=domain+'/evaluation';(TOP/'CONTROLLER_STATUS.json').write_text(json.dumps(st));run_domain();v=js(OUT/'COMPLETE.json');assert v['total_new_units']==(322 if domain=='SPARCS' else 60252);units+=v['total_new_units'];seconds+=v['GPU_seconds']
  import torch
  torch.cuda.empty_cache()
 assert units==102752 and len(SHARED_TEACHER)==535
 (TOP/'EVALUATION_COMPLETE.json').write_text(json.dumps({'units':units,'new_eval_units_including_failure':70744,'GPU_seconds_cumulative':seconds,'teacher_train_forward':0,'teacher_eval_head_forward_including_failed_cache':1070,'extra_teacher_head_recovery_calls':535,'all_source_predictions_reused_no_repeat':True}));(TOP/'EVALUATION_EXIT_CODE').write_text('0\\n')
if __name__=='__main__':main()
'''
put('evaluate_phase227_recovery03.py',t)
t=(T/'collect_phase227.py').read_text().replace('102217','102752').replace("counts['diagnostic_teacher_head']==535","counts['diagnostic_teacher_head']==1070")
t=t.replace("led=lines(e/'CALL_LEDGER.jsonl');assert [v['ordinal'] for v in led]", "led=lines(e/'CALL_LEDGER.jsonl')+(lines(e/'CALL_LEDGER_RECOVERY03.jsonl') if domain=='SPARCS' else []);assert [v['ordinal'] for v in led]")
put('collect_phase227_recovery03.py',t)
files={n:hashlib.sha256((O/'code'/n).read_bytes()).hexdigest() for n in ['evaluate_phase227_recovery03.py','collect_phase227_recovery03.py']};(O/'RECOVERY03_CODE_LOCK.json').write_text(json.dumps(files,indent=2))
with zipfile.ZipFile(O/'recovery03.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(O/'code'/n,'code/'+n)
 z.write(O/'RECOVERY03_CODE_LOCK.json','RECOVERY03_CODE_LOCK.json')
print(json.dumps({'SHA':hashlib.sha256((O/'recovery03.zip').read_bytes()).hexdigest(),'planned_actual_units':102752,'fits_repeated':0,'new_source_image_forward_repeats':0,'extra_teacher_head_calls':535}))
