"""Seal all original unlabelled scores and package only authorized shared data."""
import json,hashlib,tarfile,os,datetime
from pathlib import Path
from install_phase111_runtime import host_free
R=Path(__file__).resolve().parents[2];B=Path('/home/scv/shared/phase112_20261014');P=Path('/home/scv/shared/phase111_ocm_runtime')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 lock=json.loads((R/'src/research_results/phase112_20261014/PRE_EXECUTION_LOCK.json').read_text())
 for n,h in lock['source_sha256'].items():assert sha(R/n)==h,n
 state=json.loads((B/'run/RUN_STATUS.json').read_text());assert state['status']=='all_15_scenes_passed'
 assert state['main_calls']<=1920 and state['repeat_calls']==4 and len(state['construction_graph_calls'])<=11
 assert state['main_calls']+state['repeat_calls']+len(state['construction_graph_calls'])<=1935
 assert max(state['repeat_max_abs'])<=1e-5
 receipts=json.loads((B/'SCENE_RECEIPTS.json').read_text());assert len(receipts)==15 and {r['product_id'] for r in receipts}==set(lock['order'])
 paths={};hashes={};points=0;cpu_rows=[]
 for x in lock['products']:
  pid=x['product_id'];root=P/'single_scene_recovery_retry_20261014' if pid==lock['reuse_product'] else B/'run'/pid
  path=root/'UNLABELLED_POINT_SCORES.json';digest=sha(path)
  if pid==lock['reuse_product']:assert digest==lock['reuse_score_sha256']
  else:
   assert digest==next(r['score_sha256'] for r in receipts if r['product_id']==pid)
   cpu=json.loads((root/'CPU_OUTPUT_RECONSTRUCTION.json').read_text());assert cpu['max_absolute_error']<=1e-5 and cpu['gradient_field_max_error']==0 and cpu['coverage_exactly_equal']
  cpu=json.loads((root/'CPU_OUTPUT_RECONSTRUCTION.json').read_text());assert cpu['max_absolute_error']<=1e-5 and cpu['independent_ensemble_max_error']<=1e-5 and cpu['gradient_field_max_error']==0 and cpu['coverage_exactly_equal']
  cpu_rows.append(dict(product_id=pid,**cpu))
  hashes[pid]=digest;paths[pid]=path;points+=len(json.loads(path.read_text())['points'])
 assert points==9500
 assert Path('/home/scv/shared/phase112_run_logs_20261014/EXIT_CODE').read_text().strip()=='0'
 try:os.kill(int((B/'RUN_PID').read_text()),0)
 except ProcessLookupError:pass
 else:raise AssertionError('Worker still running')
 (B/'CPU_MOSAIC_ACCEPTANCE.json').write_text(json.dumps({'status':'passed','all16_scenes':cpu_rows,'no_extra_model_calls':True,'tolerance':1e-5},indent=2)+'\n')
 free=host_free();assert free>=4000000000
 budget=json.loads((B/'DOWNLOAD_BUDGET.json').read_text());assert budget['prior_received']+budget['received_bytes']<=4000000000
 with (B/'SCORES_SEALED.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'score_sha256':hashes,'points':points,'before_reference_join':True,'new_main_calls':state['main_calls'],'new_repeat_calls':state['repeat_calls'],'new_construction_calls':len(state['construction_graph_calls']),'host_free_after_bytes':free,'intermediate_bytes_before_package':sum(p.stat().st_size for p in B.rglob('*') if p.is_file() and 'B8A' not in p.parts),'elapsed_seconds':state['elapsed_seconds'],'new_asset_received_bytes':budget['received_bytes'],'cumulative_asset_received_bytes':budget['prior_received']+budget['received_bytes'],'training':0,'fitting':0,'threshold_scans':0,'labels_on_remote':False},f,indent=2);f.write('\n')
 with tarfile.open(B/'score_delivery_private.tar.gz','x:gz') as t:
  for pid,path in paths.items():t.add(path,arcname=pid+'/UNLABELLED_POINT_SCORES.json')
 summary={'score_archive_sha256':sha(B/'score_delivery_private.tar.gz'),'score_archive_bytes':(B/'score_delivery_private.tar.gz').stat().st_size,'seal_sha256':sha(B/'SCORES_SEALED.json'),'labels_accessed':False,'extra_model_calls':0}
 (B/'DELIVERY_RECEIPT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
