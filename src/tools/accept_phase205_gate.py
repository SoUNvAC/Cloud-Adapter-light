"""Verify and seal the returned small gate receipts locally before either fit."""
import base64,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase205/phase205_20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=O/sys.argv[1];snapshot=json.loads(p.read_text(encoding='utf-8-sig'));assert snapshot['exit_code']=='0','gate not completed successfully'
 for n,q in snapshot.items():
  if n=='exit_code':continue
  b=base64.b64decode(q['base64']);assert hashlib.sha256(b).hexdigest()==q['sha256'];dest=O/'gate_receipts'/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==b
  else:
   with dest.open('xb') as f:f.write(b)
 g=json.loads((O/'gate_receipts/gate/GATE_ACCEPTANCE.json').read_text());w=json.loads((O/'gate_receipts/gate/SMP_WHITELIST.json').read_text())
 assert g['status']=='passed' and g['model_version']==4 and g['library']=='smp' and g['classes']==['Clear','ThickCloud','ThinCloud','Shadow']
 for k in ('R4_official_vs_direct_max_error','E4_official_vs_independent_max_error','raw_mean_max_error'):assert g[k]<=1e-5
 for k in ('decoder_head_finite_nonzero_gradients','encoder_no_grad_and_unchanged','post_gate_initial_state_restored','no_training_started'):assert g[k] is True
 assert sha(O/'gate_receipts/gate/SMP_WHITELIST.json')==g['whitelist_sha256']
 assert sum(p['numel'] for p in w['parameters'] if p['trainable'])==w['trainable_parameters']
 assert all(n.startswith(('decoder.','segmentation_head.')) for n in w['trainable'])
 for seed in (12201,12301):assert sha(O/f'{seed}_PAIRED_CROP_SCHEDULE.json')==g['schedule_SHA'][str(seed)]
 lock=dict(gate_sha256=sha(O/'gate_receipts/gate/GATE_ACCEPTANCE.json'),code_sha256=json.loads((O/'CODE_LOCK.json').read_text())['files']['execute_phase205.py'],input_lock_sha256=sha(O/'INPUT_LOCK.json'),accepted_locally=True,original_schedules_verified=True,no_training_before_this_lock=True,gate_transfer_sha256=sha(p))
 target=O/'GATE_ACCEPTED_LOCAL.json'
 if target.exists():assert json.loads(target.read_text())==lock
 else:
  with target.open('x') as f:json.dump(lock,f,indent=2);f.write('\n')
 print(json.dumps(dict(status='gate_accepted_locally',trainable_parameters=w['trainable_parameters'],total_parameters=w['total_parameters'],decoder_head_BN=sum(n['name'].startswith(('decoder.','segmentation_head.')) for n in w['BN']),gate_errors={k:g[k] for k in ('R4_official_vs_direct_max_error','E4_official_vs_independent_max_error','raw_mean_max_error')})))
if __name__=='__main__':main()
