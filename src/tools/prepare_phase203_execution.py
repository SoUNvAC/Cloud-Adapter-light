"""Local Phase203 authorization/budget audit; no model construction or inference."""
import hashlib,json,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase203/phase203_20261018'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
 audit=O/'state_audit';lock=load(audit/'INPUT_CODE_LOCK.json')
 for f in lock['files']:assert sha(f['path'])==f['sha256'],f['path']
 for n,h in load(audit/'SHA256SUMS.json').items():assert sha(audit/n)==h,n
 planpath=R/'outputs/phase123/phase123_20261016/delivery/INPUT_PATCH_PLAN.json';plan=load(planpath)
 rows=[r for r in plan['rows'] if r['pool'] in ('Catalogue','PixBox')];assert len(rows)==48
 assert sum(r['pool']=='Catalogue' for r in rows)==32 and sum(r['pool']=='PixBox' for r in rows)==16
 patches={p:sum(len(r['actual_patch_coordinates']) for r in rows if r['pool']==p) for p in ('Catalogue','PixBox')}
 spot=[r for r in rows if 'T32TML_' in r['product_id'] or 'T19FDV_' in r['product_id']];assert len(spot)==2
 main=4*sum(patches.values());spotcalls=3*sum(len(r['actual_patch_coordinates']) for r in spot);constructor=7*7;warm=7*2;planned=main+spotcalls+constructor+warm
 assert main==4584 and spotcalls==204 and planned==4851 and planned<=5000
 white=load(R/'src/research_results/phase122_20261015/C100_PARAMETER_WHITELIST.json')
 params=set(white['trainable_names']);buffers=set(white['decoder_state_names'])-params
 assert len(params)==46 and len(buffers)==15 and all(n.endswith(('.running_mean','.running_var','.num_batches_tracked')) for n in buffers)
 report=dict(status='local_preflight_passed_remote_not_started',date_Beijing='2026-10-19',protocol_sha256=sha(R/'src/research_plans/PHASE203_DECODER_PARAMETER_BN_MECHANISM_20261018.md'),state_audit_verified=True,plan_source_sha256=sha(planpath),patches_by_pool=patches,new_combination_calls=main,reconstruction_spot_calls=spotcalls,construction_allowance=constructor,warmup_allowance=warm,planned_total=planned,approved_total_cap=5000,engineering_reserve=5000-planned,original_cap=4000,authorization='Human explicitly allowed cumulative5000 calls, all controls retained; other limits unchanged',new_states=['12201_10','12201_01','12301_10','12301_01'],reconstruction_states=['shared_R_00','12201_11','12301_11'],spot_products=[r['product_id'] for r in spot],fixed_threshold=.7693686485290528,GPU_stage_seconds_cap=7200,GPU_bytes_cap=20*1024**3,CPU_bytes_cap=16*1024**3,new_persistent_bytes_cap=6_000_000_000,host_free_bytes_floor=4_000_000_000,local_new_neural_calls=0,local_new_fits=0,decoder_parameters=sorted(params),decoder_BN_buffers=sorted(buffers),no_calibration_input_read=True,code_sha256=sha(Path(__file__)))
 save(O/'PRE_EXECUTION_BUDGET.json',report)
 save(O/'EVALUATION_INPUT_PLAN.json',dict(rows=rows,original_fixed_patch_coordinates=True,calibration_rows_excluded=True,source_sha256=sha(planpath)))
 save(O/'EXECUTION_STATUS.json',dict(status='blocked_ssh_before_inference',new_calls=0,new_fits=0,remote_inventory_completed=False,ssh_attempts=2,ssh_error='kex_exchange_identification: Connection closed by remote host',authorization_5000_recorded=True))
 print(json.dumps({k:report[k] for k in ('status','patches_by_pool','new_combination_calls','reconstruction_spot_calls','planned_total','approved_total_cap','engineering_reserve')}))
if __name__=='__main__':main()
