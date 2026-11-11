from pathlib import Path
import json,hashlib,torch
R=Path('/home/scv/shared/phase477_20261107');D=R/'A/network_run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('/home/scv/shared/phase431_20261102/fused/best_mIoU_iter_32000.pth');bp=Path('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth')
assert sha(source)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
assert sha(bp)=='3051e08ea9733c82202e1425ffc52046132b149bd62a157e863ed156c95fc533'
ss=torch.load(source,map_location='cpu')['state_dict'];bs=torch.load(bp,map_location='cpu')['state_dict'];origin=torch.load(D/'branch_0.pth',map_location='cpu')
assert set(origin['state_dict'])==set(bs)
assert all(torch.equal(t,origin['state_dict'][n]) for n,t in bs.items())
assert all(torch.equal(t,ss[n]) for n,t in origin['student_patch'].items())
lock=json.loads((D/'LOCK.json').read_text());assert lock['task_parameters']>0 and lock['new_module'] is None
assert all(n.startswith('decode_head.') for n in lock['active_task_names'])
final=json.loads((D/'FINAL_RESULTS.json').read_text());weights={}
for step in sorted(set(final['choices'].values())|{0}):
 p=D/f'branch_{step}.pth';state=torch.load(p,map_location='cpu')
 assert set(state['state_dict'])==set(bs) and all(torch.equal(t,state['state_dict'][n]) for n,t in bs.items())
 changed={n:not torch.equal(t,state['student_patch'][n]) for n,t in origin['student_patch'].items()}
 assert all(not v for n,v in changed.items() if not n.startswith('decode_head.'))
 if step:assert any(changed.values())
 weights[p.name]=dict(sha256=sha(p),changed_decoder_tensors=sum(changed.values()))
for arm in ['A','B']:
 d=Path('/home/scv/shared/phase476_20261107')/arm/'network_run';old=torch.load(d/'branch_0.pth',map_location='cpu')
 assert all(torch.equal(t,old['student_patch'][n]) for n,t in origin['student_patch'].items())
 assert all(torch.equal(t,old['state_dict'][n]) for n,t in bs.items())
 assert (D/'INPUT_TRACE.jsonl').read_bytes()==(d/'INPUT_TRACE.jsonl').read_bytes()
result=dict(passed=True,source_sha=sha(source),origin450_sha=sha(bp),task_parameters=lock['task_parameters'],new_parameters=0,only_decoder_updated=True,original450_branch_fixed=True,matches_476_common_initialization_and_recorded_batches=True,batch_trace_steps=[json.loads(x)['step'] for x in (D/'INPUT_TRACE.jsonl').read_text().splitlines()],batch_check_scope='Same seed/sampler and byte-identical recorded signatures at listed steps',weights=weights)
(R/'TRAINED_WEIGHT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print('TRAINED_WEIGHT_ACCEPTANCE passed')
