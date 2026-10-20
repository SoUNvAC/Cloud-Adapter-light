"""Non-neural Phase300 admission in the explicitly requested conda environment."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Optional

sys.dont_write_bytecode = True
REPO = Path('/home/scv/Cloud-Adapter-light')
ROOT = Path('/home/scv/shared/phase300_20261020')


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def function(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def main():
    import numpy as np
    import torch
    from phase300_images import load_image, training_access_guard
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'REMOTE_PREFLIGHT.json').exists():
        print((ROOT / 'REMOTE_PREFLIGHT.json').read_text())
        return
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'cloud-lite-pt210'
    versions = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                    cuda_available=torch.cuda.is_available(),
                    dependencies={k:bool(importlib.util.find_spec(k)) for k in
                                  ('segmentation_models_pytorch', 'timm', 'safetensors', 'rasterio')})
    if torch.cuda.is_available():
        versions.update(device=torch.cuda.get_device_name(), cuda_free_total=list(torch.cuda.mem_get_info()))
    write('ENVIRONMENT.json', versions)
    manifest_path = REPO / 'src/research_results/phase300_20261020/FIT_IMAGES.json'
    rows = json.loads(manifest_path.read_text())['rows']
    assert len(rows) == len({r['group_id'] for r in rows}) == 100
    vendor = Path('/home/scv/shared/phase111_ocm_runtime/assets/vendor/omnicloudmask')
    expected = {'model_utils.py':'f749ceced40f8b0899f56e95e79b9f43ea77b74898376175407c21c26796a1dd',
                'cloud_mask.py':'9130116e5daa130ceaed28de222484f141abf365c614c0b77aaf7b4f8cf7c33c'}
    actual = {name:hashlib.sha256((vendor/name).read_bytes()).hexdigest() for name in expected}
    assert actual == expected, 'Historical vendor changed'
    scope = dict(np=np, Optional=Optional)
    function(vendor/'model_utils.py', ['channel_norm'], scope)
    function(vendor/'raster_utils.py', ['make_patch_indexes', 'get_patch'], scope)
    raster_sha = hashlib.sha256((vendor/'raster_utils.py').read_bytes()).hexdigest()
    # The guard is installed before opening any fit image; forbidden-label
    # access is deliberately attempted and must raise PermissionError.
    audit = training_access_guard(rows[0]['image_archive'], ROOT/'FILE_ACCESS.jsonl')
    denied = False
    try:
        with (Path(rows[0]['image_archive']).parent/'masks.zip').open('rb'):
            pass
    except PermissionError:
        denied = True
    assert denied
    records = []
    start = time.monotonic()
    for row in rows:
        image, valid = load_image(row, ROOT/'IMAGE_ACCESS.jsonl')
        assert image.shape == (3, 1022, 1022) and np.isfinite(image).all()
        windows = []
        for index in scope['make_patch_indexes'](1022, 1022, 1000, 300):
            patch, moved = scope['get_patch'](image, index, 0)
            if patch is not None and list(moved) not in windows:
                assert patch.shape == (3,1000,1000) and np.isfinite(patch).all()
                windows.append(list(moved))
        records.append(dict(product=row['product'], group_id=row['group_id'],
                            image_valid_pixels=int(valid.sum()), actual_windows=windows))
        write('PROGRESS.json', dict(products=len(records), neural_forward_calls=0))
        del image, valid
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 16*1024**3
    write('FIT_WINDOWS.json', dict(rows=records, raster_utils_sha256=raster_sha,
                                 vendor_sha256=actual))
    # The crop sequence uses only dimensions and product indexes. It never
    # consults labels, confidence, or image-valid counts.
    rng = np.random.RandomState(30001)
    schedule = [dict(attempt=i+1, fit_index=int(rng.randint(100)),
                     top=int(rng.randint(511)), left=int(rng.randint(511)),
                     horizontal_flip=bool(rng.rand()<0.5)) for i in range(1000)]
    write('CROP_SCHEDULE.json', dict(seed=30001, rows=schedule,
                                   shared_by=['B-PL','M-PL'], retry_empty=False))
    T = sum(len(r['actual_windows']) for r in records)
    missing = [k for k,v in versions['dependencies'].items() if not v]
    blockers = []
    if missing: blockers.append('Missing dependencies in requested conda: ' + ', '.join(missing))
    if versions['torch'] != '2.11.0+cu128':
        blockers.append('Requested conda torch differs from locked Phase205/208 runtime 2.11.0+cu128')
    if not versions['cuda_available']: blockers.append('CUDA unavailable')
    write('REMOTE_PREFLIGHT.json', dict(status='blocked' if blockers else 'needs_model_acceptance',
          blockers=blockers, teacher_actual_T=T, training_max=2000, evaluation=2292,
          probes=32, planned_main_calls=T+2000+2292+32,
          construction_acceptance_recovery_margin=7000-(T+2000+2292+32),
          neural_forward_calls=0, optimizer_updates=0, fit_products=len(records),
          label_open_denied=denied, all_fit_loads_succeeded_with_labels_denied=True,
          access_audit_scope='Python audit events; not native syscall tracing; no training acceptance claimed',
          SHA_verified_image_members=len(records), seconds=time.monotonic()-start,
          rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          GPU_stage_started=False, resource_time_projection_not_yet_measured=True))
    audit.flush()
    print((ROOT/'REMOTE_PREFLIGHT.json').read_text())


if __name__ == '__main__':
    main()
