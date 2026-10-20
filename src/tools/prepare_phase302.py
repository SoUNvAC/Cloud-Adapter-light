"""Read-only Source/data admission; no source pixels, labels or model forward."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone, timedelta

R = Path(__file__).resolve().parents[2]
ROOT = Path('/home/scv/shared/phase302_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d'))
EXPECTED = '64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, value): Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    assert R == Path('/home/scv/Cloud-Adapter-light')
    assert not ROOT.exists(), 'Preserve existing attempt'
    ROOT.mkdir()
    for name in ('tmp', 'cache', 'code'): (ROOT / name).mkdir()
    os.environ.update(TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'),
        MPLCONFIGDIR=str(ROOT/'cache'), HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    evidence = {}
    def pin(p):
        p = Path(p); evidence[str(p)] = sha(p); return load(p)
    source = R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
    assert sha(source) == EXPECTED
    fit_path = R/'src/research_results/phase300_20261020/FIT_IMAGES.json'
    fit = pin(fit_path)['rows']; assert len(fit) == 100
    assert all(set(row) == {'product','group_id','image_archive','image_member','image_member_sha256','shape'} for row in fit)
    old = Path('/home/scv/shared/phase300_20261020')
    evaluation = [r for r in pin(old/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue']
    assert len(evaluation) == 32 and not ({r['product'] for r in fit} & {r['product_id'] for r in evaluation})
    schedule = pin(old/'CROP_SCHEDULE.json'); assert len(schedule['rows']) == 1000
    assert [r['attempt'] for r in schedule['rows']] == list(range(1,1001))
    for row in schedule['rows']:
        assert 0 <= row['fit_index'] < 100 and 0 <= row['top'] <= 510 and 0 <= row['left'] <= 510
    provenance = R/'src/work_dirs/phase65a/source_provenance'
    spatial = pin(provenance/'spatial_overlap.json')
    mapping_path = R/'src/work_dirs/phase65a/source_mapping/mapping_audit.json'
    mapping = pin(mapping_path)
    assert mapping['status'] == 'verified_rgb_ordinal_mapping'
    assert sha(mapping_path) == spatial['source_mapping_sha256']
    assert mapping['splits']['train']['rows'] == 8490 and mapping['splits']['val']['rows'] == 535
    findings = {r['product']: r for r in spatial['findings']}
    selected = [r['product'] for r in fit] + [r['product_id'] for r in evaluation]
    assert all(p in findings for p in selected), 'Source spatial audit missing selected products'
    overlaps = [p for p in selected if findings[p]['source_footprint_overlap']]
    assert not overlaps, 'Known Source footprint overlap: ' + str(overlaps)
    original_cfg = R/'src/work_dirs/phase64_source_parent/seed64/phase64_source_parent_rgb.py'
    evidence[str(original_cfg)] = sha(original_cfg)
    sys.path.insert(0, str(R/'src'))
    import torch, mmcv, mmseg
    from mmcv.ops import MultiScaleDeformableAttention
    from mmengine.config import Config
    assert Path(sys.executable).resolve() == Path('/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python').resolve()
    assert torch.__version__ == '2.1.0+cu121' and torch.cuda.is_available()
    cfg = Config.fromfile(str(R/'src/configs/protocol/phase64_source_parent_rgb.py'))
    original = Config.fromfile(str(original_cfg))
    assert cfg.model == original.model, 'Historical Source model config drift'
    assert cfg.train_dataloader.dataset.data_prefix.img_path == 'img_dir/train'
    assert cfg.val_dataloader.dataset.data_prefix.img_path == 'img_dir/val'
    assert cfg.train_cfg.max_iters == 4000 and cfg.default_hooks.checkpoint.save_best == 'mIoU'
    cfg.dump(str(ROOT/'resolved_source_config.py'))
    checkpoint = torch.load(source, map_location='cpu')
    meta = checkpoint.get('meta', {})
    assert meta['iter'] == 4000, 'Unexpected Source checkpoint iteration'
    write(ROOT/'SOURCE_ADMISSION.json', dict(status='passed', source_sha256=EXPECTED,
        runtime=sys.executable, torch=torch.__version__, mmcv=mmcv.__version__, mmseg=mmseg.__version__,
        gpu=torch.cuda.get_device_name(), cuda_free_total=torch.cuda.mem_get_info(),
        source_training='CloudSEN12-high L1C official train8490, val535; seed64; 4000 steps; best source-val mIoU',
        classes=['Surface','Cloud','Shadow'], selected_source_spatial_overlaps=overlaps,
        source_mapping_status=mapping['status'], spatial_buffer_m=spatial['native_buffer_m'],
        history='Catalogue development used repeatedly in earlier phases; fit identities selected historically with label metadata. Not new blind evaluation.',
        unresolved='DINO pretraining intersection unknown; Source mapping/audits inherited, not reopened source rasters; metadata cannot prove all pretraining independence.',
        evidence_sha256=evidence, source_meta_keys=sorted(meta), neural_calls=0))
    # No truth fields enter the GPU worker manifest.
    evaluation_images = [dict(product=r['product_id'], group_id=r['group']['group_id'],
        image_archive=fit[0]['image_archive'], image_member='subscenes/'+r['product_id']+'.npy',
        image_member_sha256=r['group']['image_member_sha256'], shape=[1022,1022,13]) for r in evaluation]
    write(ROOT/'FIT_IMAGES.json', dict(rows=fit))
    write(ROOT/'EVAL_IMAGES.json', dict(rows=evaluation_images))
    shutil.copyfile(old/'CROP_SCHEDULE.json', ROOT/'CROP_SCHEDULE.json')
    write(ROOT/'PROTOCOL_LOCK.json', dict(status='admitted', fit_count=100, evaluation_count=32,
        source_sha256=EXPECTED, input_files={n:sha(ROOT/n) for n in ('FIT_IMAGES.json','EVAL_IMAGES.json','CROP_SCHEDULE.json')},
        protocol_sha256=sha(R/'src/research_plans/phase302.md'),
        code={str(p.relative_to(R)):sha(p) for p in sorted((R/'src/tools').glob('*phase302*')) if p.is_file()},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        source_audit_sha256=sha(ROOT/'SOURCE_ADMISSION.json'), seed=30201,
        cap_calls=4000, cap_seconds=14400, cap_gpu_bytes=20*1024**3, cap_cpu_bytes=16*1024**3,
        cap_new_bytes=6000000000, min_available_bytes=4000000000, neural_calls=0))
    print(json.dumps(dict(status='admitted',root=str(ROOT),neural_calls=0)))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        if ROOT.exists(): write(ROOT/'ADMISSION_FAILURE.json', dict(status='failed',error=str(exc),traceback=traceback.format_exc(),neural_calls=0))
        raise
