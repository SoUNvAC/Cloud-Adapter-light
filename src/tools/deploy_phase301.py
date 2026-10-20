"""Guarded zero-neural CPU launch inside the allowed Phase301 shared directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

R=Path('/home/scv/Cloud-Adapter-light')
ROOT=Path('/home/scv/shared/phase301_20261020')
PYTHON=Path('/home/scv/shared/phase111_ocm_runtime/runtime_cuda128/bin/python')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (ROOT/'DISPATCH.json').exists():
        print((ROOT/'DISPATCH.json').read_text());return
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    with zipfile.ZipFile(ROOT/'inputs.zip') as archive:
        assert set(archive.namelist())=={'deduplicated_locations.json','INPUT_LOCK.json'}
        lock=json.loads(archive.read('INPUT_LOCK.json'))
        content=archive.read('deduplicated_locations.json')
    assert hashlib.sha256(content).hexdigest()==lock['PixBox_reference_sha256']=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'
    pix=ROOT/'inputs/data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    pix.parent.mkdir(parents=True,exist_ok=True);pix.write_bytes(content)
    source=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871/masks.zip')
    assert sha(source)==lock['Catalogue_masks_sha256']
    target=ROOT/'inputs/data/sentinel2_cloud_mask_catalogue_4172871/masks.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): assert sha(target)==sha(source)
    else:
        try: target.hardlink_to(source)
        except OSError: shutil.copyfile(source,target)
    code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase301*.py')}
    (ROOT/'REMOTE_INPUT_LOCK.json').write_text(json.dumps(dict(code=code,inputs=lock,
       protocol_sha256=sha(R/'src/research_plans/phase301.md'),neural_calls=0),indent=2)+'\n')
    for key in ('XDG_CACHE_HOME','MPLCONFIGDIR','TMPDIR','HF_HOME','TORCH_HOME'):
        directory=ROOT/'controlled_caches'/key.lower();directory.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(directory)
    os.environ.update(PHASE301_REMOTE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
                      OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (ROOT/'controller.log').open('xb') as stream:
        process=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/run_phase301.py')],cwd=R,
            stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record=dict(PID=process.pid,code=code,neural_calls_planned=0,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip())
    (ROOT/'DISPATCH.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(PID=process.pid,neural_calls_planned=0)))


if __name__=='__main__': main()
