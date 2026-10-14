"""Seal the approved runtime overlay before assets/inference; no model imports."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import argparse

REPO=Path(__file__).resolve().parents[2]
BASE=Path('/home/scv/shared/phase111_ocm_runtime')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authorized-recovery',action='store_true');ap.add_argument('--authorized-construction-retry',action='store_true');args=ap.parse_args()
    runtime=json.loads((BASE/'RUNTIME_IMPORT_ACCEPTANCE.json').read_text())
    assert runtime['dependency_check']=='passed' and runtime['cuda_available']
    original=REPO/'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json'
    runner=REPO/'src/tools/run_phase111_acceptance.py'
    addon=REPO/'src/research_plans/PHASE111_ENVIRONMENT_RESUME_20261014.md'
    paths=[original,runner,addon,BASE/'resource_gate_20261014_cuda128_metadata/PIP_RESOLUTION.json']
    obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
         'runtime':runtime,'environment_4GB_cap_user_override':True,
         'physical_reserve_bytes':4000000000,'ram_bytes':16*1024**3,'vram_bytes':20*1024**3,
         'max_gpu_phase_wall_seconds':3600,'max_model_call_attempts_including_probes':256,
         'main_scene_attempts':1,'repeat_first_valid_patches':2,'extra_repeat_calls':4,
         'repeat_max_abs_tolerance':1e-5,'cpu_reconstruction_max_abs_tolerance':1e-5,
         'input_mask':'native specials; 1-native-pixel square support halo for B04/B03 bilinear; native B8A',
         'model_calls_before_lock':0,'no_labels_or_performance_metrics':True}
    destination=BASE/'PRE_FORWARD_LOCK.json'
    if args.authorized_recovery or args.authorized_construction_retry:
        old=BASE/'single_scene_acceptance/RUN_STATUS.json'
        prior=json.loads(old.read_text())
        assert prior['status']=='failed_no_automatic_rerun'
        assert prior['main_calls']==128 and prior['repeat_calls']==4 and len(prior['construction_graph_calls'])==11
        assert json.loads((BASE/'GRADIENT_VERIFIER_FIX_VERIFIED.json').read_text())['fixed_independent_cpu_gradient_field_max_abs']==0
        recovery_protocol=REPO/'src/research_plans/PHASE111_CPU_VERIFIER_RECOVERY_20261014.md'
        obj['source_sha256'][str(recovery_protocol)]=hashlib.sha256(recovery_protocol.read_bytes()).hexdigest()
        obj.update(previous_model_call_attempts=143,max_model_call_attempts_including_probes=286,
                   max_gpu_phase_wall_seconds=3400,model_calls_before_lock=143,
                   authorized_recovery=True,prior_attempt_directory=str(old.parent),
                   prior_failed_status_sha256=hashlib.sha256(old.read_bytes()).hexdigest())
        destination=BASE/'PRE_FORWARD_RECOVERY_LOCK.json'
    if args.authorized_construction_retry:
        failed=BASE/'single_scene_recovery_20261014/RUN_STATUS.json'
        second=json.loads(failed.read_text())
        assert second['status']=='failed_no_automatic_rerun'
        assert second['main_calls']==0 and second['repeat_calls']==0 and len(second['construction_graph_calls'])==11
        assert second['previous_model_call_attempts']==143
        obj.update(previous_model_call_attempts=154,model_calls_before_lock=154,
                   max_model_call_attempts_including_probes=297,
                   interrupted_recovery_status_sha256=hashlib.sha256(failed.read_bytes()).hexdigest(),
                   authorized_construction_retry=True)
        destination=BASE/'PRE_FORWARD_RECOVERY_RETRY_LOCK.json'
    with destination.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
    print(json.dumps({'status':'pre_forward_lock_saved','git_commit':obj['git_commit']}))


if __name__=='__main__':main()
