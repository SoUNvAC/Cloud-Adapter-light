"""Read-only historical protocol evidence audit; no model execution or training."""
from pathlib import Path
import json,csv,hashlib,math,collections
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/audit_phase400_427/20261101'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def main():
 OUT.mkdir(parents=True,exist_ok=True);inventory=[];runs=[];evidence={};integrity=[];receipts=[]
 reference=ROOT/'outputs/phase411/phase411_20261101/layerwise/formal_STEP_TRACE.jsonl'
 ref=[json.loads(x) for x in reference.read_text().splitlines()]
 reference_fields=['step','draw','product','image_SHA','label_SHA','augmentation_seed']
 for phase in range(400,428):
  roots=[p.parent for p in (ROOT/f'outputs/phase{phase}').rglob('RESULT.md') if 'attempt' not in str(p)]
  assert len(roots)==1,(phase,roots)
  r=roots[0];plan=ROOT/f'src/research_plans/phase{phase}.md'
  inventory.append({'phase':phase,'root':rel(r),'result':rel(r/'RESULT.md'),'plan':rel(plan)})
  for p in [plan,r/'RESULT.md']:evidence[rel(p)]=sha(p)
  for p in r.rglob('*.json'):
   if any(t in p.name for t in ['ACCEPTANCE','SELF_TEST','LOCK','FREEZE','SCOPE','COMPLETE','RECOVERY','RETIRED']):
    evidence[rel(p)]=sha(p)
    if 'ACCEPTANCE' in p.name:
     v=js(p);receipts.append({'phase':phase,'path':rel(p),'passed':v.get('passed'),'retired_attempts':v.get('retired_attempts',[])})
  manifests=[p for p in r.glob('*MANIFEST.json') if p.name in ('LOCAL_MANIFEST.json','TRANSFER_MANIFEST.json')]
  manifests += list((r/'delivery').glob('SHA256SUMS.json')) if (r/'delivery').exists() else []
  for p in manifests:
   data=js(p)
   if not isinstance(data,dict):continue
   pairs=[(n,h) for n,h in data.items() if isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)]
   if not pairs:continue
   missing=[];mismatch=[];checked=0
   for name,h in pairs:
    target=p.parent/name
    if not target.exists():missing.append(name);continue
    checked+=1
    if sha(target)!=h:mismatch.append(name)
   integrity.append({'phase':phase,'manifest':rel(p),'checked':checked,'missing':missing,'mismatch':mismatch})
  traces=[p for p in r.rglob('*STEP_TRACE.jsonl') if 'short' not in p.name and '/short/' not in p.as_posix()]
  for p in traces:
   rows=[json.loads(x) for x in p.read_text().splitlines()];assert rows
   scope=p.with_name(p.name.replace('STEP_TRACE.jsonl','ACTIVE_SCOPE.json'))
   v=js(scope) if scope.exists() else {};names=v.get('names',{})
   count=sum(names.values())
   assert not names or count==v.get('count',count)
   def lrs(row):
    value=row.get('LR',[]);return sorted(set(value if isinstance(value,list) else [value]))
   paired=None
   if phase>=409 and all(k in rows[0] for k in reference_fields):
    paired=len(rows)<=len(ref) and all(all(a[k]==b[k] for k in reference_fields) and lrs(a)==lrs(b) for a,b in zip(rows,ref))
   losses=sorted(set(k for a in rows for k in a.get('losses',{})))
   finite=all(math.isfinite(float(v)) for a in rows for v in list(a.get('losses',{}).values())+([a['loss']] if 'loss' in a else []))
   runs.append({'phase':phase,'trace':rel(p),'steps':len(rows),'continuous_steps':all(a['step']==i+1 for i,a in enumerate(rows)),
                'trainable_count':v.get('count',count or None),'trainable_families':';'.join(sorted(set('.'.join(n.split('.')[:3]) for n in names))),
                'initial_LR':str(lrs(rows[0])),'final_LR':str(lrs(rows[-1])),
                'paired_inputs_and_LR_vs411':paired,'finite_logged_losses':finite,'loss_keys':';'.join(losses),
                'common_MsRE_SHA':v.get('common_MsRE_SHA',v.get('MsRE_initial_SHA'))})
   evidence[rel(p)]=sha(p)
  for p in r.rglob('*.py'):evidence[rel(p)]=sha(p)
 h_pairs=[]
 for phase in [406,408]:
  for seed in [40501,40502]:
   a=ROOT/f'outputs/phase{phase}/phase{phase}_20261101/{seed}/formal_STEP_TRACE.jsonl'
   b=ROOT/f'outputs/phase405/phase405_20261101/{seed}/formal_STEP_TRACE.jsonl'
   aa=[json.loads(x) for x in a.read_text().splitlines()];bb=[json.loads(x) for x in b.read_text().splitlines()]
   passed=len(aa)==len(bb)==4000 and all(all(x[k]==y[k] for k in reference_fields+['LR']) for x,y in zip(aa,bb))
   h_pairs.append({'phase':phase,'seed':seed,'paired_to405':passed})
 for p in (ROOT/'src/tools').glob('*.py'):
  if any(f'phase{i}' in p.name for i in range(400,428)):
   evidence[rel(p)]=sha(p)
 # Verify saved scopes, traces and receipts, without treating historical receipts as a new GPU audit.
 for name,data in [('INVENTORY.json',inventory),('RUNS.json',runs),('INTEGRITY.json',integrity),('RECEIPTS.json',receipts),('EVIDENCE_SHA256.json',evidence)]:
  (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 with (OUT/'TRAINING_RUNS.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)
 summary={'phases':len(inventory),'formal_trace_files':len(runs),'formal_trace_rows':sum(x['steps'] for x in runs),
          'noncontinuous':[x['trace'] for x in runs if not x['continuous_steps']],
          'nonfinite':[x['trace'] for x in runs if not x['finite_logged_losses']],
          'paired_failures':[x['trace'] for x in runs if x['paired_inputs_and_LR_vs411'] is False],
          'manifest_entries_checked':sum(x['checked'] for x in integrity),'manifest_mismatches':[x for x in integrity if x['mismatch']],
          'manifest_missing':[x for x in integrity if x['missing']],
          'H_pairs_vs405':h_pairs,'new_training_updates':0,'new_model_forwards':0,'scope':'local text/code/trace/hash audit; remote checkpoint tensors and pixel arrays not reloaded'}
 (OUT/'AUDIT_CHECKS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 print('RUN COUNTS',dict(collections.Counter(x['phase'] for x in runs)))
if __name__=='__main__':main()
