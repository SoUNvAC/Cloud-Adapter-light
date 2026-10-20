"""Admit and launch only the committed Phase300 implementation, once."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path('/home/scv/Cloud-Adapter-light')
ROOT = Path('/home/scv/shared/phase300_20261020')
OLD = Path('/home/scv/shared/phase208_20261020')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    if (ROOT/'LAUNCH.json').exists():
        print((ROOT/'LAUNCH.json').read_text())
        return
    assert load(ROOT/'REMOTE_PREFLIGHT.json')['teacher_actual_T'] == 400
    assert Path(sys.executable).resolve() == Path('/home/scv/shared/phase111_ocm_runtime/runtime_cuda128/bin/python').resolve()
    # Every protected historical artifact is copied byte for byte.
    for name in ('V4_WEIGHT_LOCK.json','EVALUATION_INPUT_PLAN.json','EXPECTED_SUPPORT.json','INITIALIZATION_RULE.json'):
        src = OLD/name
        assert sha(src) == load(OLD/'INPUT_LOCK.json')['files'][name]
        dst = ROOT/name
        if dst.exists(): assert sha(dst) == sha(src)
        else: shutil.copyfile(src,dst)
    for row in load(ROOT/'V4_WEIGHT_LOCK.json')['weights']:
        assert sha(Path('/home/scv/shared/phase200_20261017/assets')/row['filename']) == row['sha256']
    for filename,expected in load(OLD/'CODE_LOCK.json')['files'].items():
        assert sha(OLD/'code'/filename) == expected
    files = ['src/tools/execute_phase300.py','src/tools/deploy_phase300.py','src/tools/phase300_images.py']
    # Hash canonical committed bytes, also ensuring remote working files match.
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for name in files:
        committed = subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)
        assert hashlib.sha256(committed).hexdigest() == sha(REPO/name)
    old_times = {s:load(OLD/s/'RUN_STATUS.json')['stage_seconds'] for s in ('gate','train_B','train_M','infer')}
    time_upper = 2*sum(old_times.values()) + 1800
    assert time_upper < 14400, 'Historical timings project over 4h; report gap first'
    lock = dict(code={name:sha(REPO/name) for name in files},commit=commit,
                protocol_sha256=sha(REPO/'src/research_plans/phase300.md'),
                crop_sha256=sha(ROOT/'CROP_SCHEDULE.json'),
                manifest_sha256=sha(REPO/'src/research_results/phase300_20261020/FIT_IMAGES.json'),
                user_runtime_override='Explicit user reply permits historical fixed runtime',
                model_source='immutable Phase208 code snapshot; no trained adapters loaded',
                teacher_calls=400,training_max=2000,evaluation=2292,probes=32,
                planned_main_calls=4724,acceptance_calls=5,constructor_recovery_margin=2271,
                cap_calls=7000,cap_seconds=14400,historical_stage_seconds=old_times,
                conservative_time_projection_seconds=time_upper,projection_is_not_measurement=True)
    (ROOT/'EXECUTION_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
    with (ROOT/'pipeline.console.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools/execute_phase300.py'),'run'],
                    cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (ROOT/'DISPATCH.json').write_text(json.dumps(dict(PID=process.pid,epoch=time.time(),commit=commit)))
    print(json.dumps(dict(PID=process.pid,commit=commit,planned_main_calls=4724,time_upper=time_upper)))


if __name__ == '__main__':
    main()
