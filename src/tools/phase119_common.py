"""Shared fixed input/identity helpers; no model or label-driven selection."""
import io
import os
from pathlib import Path
import sys
import zipfile
from phase118_common import sha, load, write, ram, coordinates, image, BASE, REPO
import phase118_common as p118

ROOT = Path('/home/scv/shared/phase119_20261015')
CAT = Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
OLD = Path('/home/scv/shared/phase117_20261014')
LOCK = REPO/'src/research_results/phase119_20261015/PRE_EXECUTION_LOCK.json'


def setup():
    cfg = load(LOCK)
    for p,h in cfg['source_sha256'].items():
        assert sha(REPO/p)==h,p
    for key in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH',
                'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR'):
        os.environ[key]=str(BASE/'controlled_caches'/key.lower())
    os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',GDAL_NUM_THREADS='1',OPJ_NUM_THREADS='1')
    sys.path.insert(0,str(BASE/'assets/vendor'))
    return cfg


def catalog_image(group, read_truth=False, hash_seconds=None):
    import numpy as np
    pid=group['product']; member=f'subscenes/{pid}.npy'
    with zipfile.ZipFile(CAT/'subscenes.zip') as z:
        raw_bytes=z.read(member)
    with zipfile.ZipFile(CAT/'masks.zip') as z:
        mask_bytes=z.read(f'masks/{pid}.npy')
    import hashlib
    import time
    start=time.perf_counter()
    assert hashlib.sha256(raw_bytes).hexdigest()==group['image_member_sha256']
    assert hashlib.sha256(mask_bytes).hexdigest()==group['mask_member_sha256']
    if hash_seconds is not None:hash_seconds[0]+=time.perf_counter()-start
    raw=np.load(io.BytesIO(raw_bytes),allow_pickle=False)
    mask=np.load(io.BytesIO(mask_bytes),allow_pickle=False)
    assert raw.shape==(1022,1022,13) and raw.dtype==np.float32
    assert mask.shape==(1022,1022,3) and mask.dtype==np.bool_
    observed=np.isfinite(raw)&(raw!=0)
    good=observed&(raw!=np.float32(65535/10000))
    valid=(mask.sum(-1)==1)&observed[...,[3,2,1,7,11,12]].all(-1)&good[...,[3,2,8,1,11,12]].all(-1)
    assert int(valid.sum())==group['common_valid']
    arr=raw[...,[3,2,8]].transpose(2,0,1).copy()
    arr[~good[...,[3,2,8]].transpose(2,0,1)]=0
    truth=mask.argmax(-1)[valid].astype(np.uint8) if read_truth else None
    return arr,valid,truth


def original_pix_score(pid):
    cfg=load(REPO/'src/research_results/phase112_20261014/PRE_EXECUTION_LOCK.json')
    root=BASE/'single_scene_recovery_retry_20261014' if pid==cfg['reuse_product'] else Path('/home/scv/shared/phase112_20261014/run')/pid
    p=root/'UNLABELLED_POINT_SCORES.json'
    expected=load(REPO/'src/research_results/phase112_20261014/SCORES_SEALED.json')['score_sha256'][pid]
    assert sha(p)==expected
    return load(p),expected


def verify_native_and_cache():
    """One content hash pass outside timing, with stat-bound reuse in timed reads."""
    files=load(REPO/'src/research_results/phase118_20261014/NATIVE_ASSET_SHA_LOCK.json')['files']
    verified={}
    b8=load(REPO/'src/research_results/phase112_20261014/DOWNLOAD_STATUS.json')['files']
    for key,v in files.items():
        pid,band=key.rsplit('/',1)
        p=(Path(b8[pid]['path']) if pid in b8 else BASE/'assets/B8A.jp2') if band=='B8A' else p118.RAW/'imagery'/pid/(band+'.jp2')
        assert sha(p)==v['sha256']
        s=p.stat();verified[str(p)]=(v['sha256'],(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns))
    def bound_sha(path):
        p=Path(path);digest,stat=verified[str(p)];s=p.stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==stat,'input changed after pre-timing hash'
        return digest
    p118.sha=bound_sha
    return len(verified)


def plan_indexes(array):
    import numpy as np
    from omnicloudmask import cloud_mask as cm,raster_utils as ru
    assert cm.check_patch_size(array,0,1000,300)==(300,1000)
    actual=[];seen=set()
    for idx in ru.make_patch_indexes(array.shape[2],array.shape[1],1000,300):
        patch,moved=ru.get_patch(array,idx,0)
        if patch is not None and tuple(moved) not in seen:
            assert patch.shape==(3,1000,1000) and np.isfinite(patch).all()
            actual.append(list(moved));seen.add(tuple(moved))
    return actual


def prepare():
    import gc
    import resource
    from install_phase111_runtime import host_free
    cfg=setup();ROOT.mkdir(exist_ok=False)
    (ROOT/'PREPARE_PID').write_text(str(os.getpid()))
    assert host_free()>=4000000000
    coord=coordinates();verify_native_and_cache()
    products=load(REPO/'src/research_results/phase104_20261013/object_manifest.json')['products']
    rows=[];old_cache={}
    for product in products:
        pid=product['product_id'];q=[v for v in coord if v['product_id']==pid]
        array,valid,bindings=image(product,'C20',q)
        actual=plan_indexes(array);_,oldhash=original_pix_score(pid)
        rows.append({'pool':'PixBox','product_id':pid,'product':product,'actual_patch_coordinates':actual,
                     'input_bindings':bindings,'old_score_sha256':oldhash,'points':len(q)})
        del array,valid;gc.collect()
        write(ROOT/'PREPARE_PROGRESS.json',{'scenes':len(rows),'model_calls':0,'last':pid})
    dev=[g for g in load(REPO/'src/research_results/phase117_20261014/INPUT_GATE.json')['bindings'] if g['split']=='development_val']
    assert len(dev)==32 and len({g['group_id'] for g in dev})==32
    split=REPO/'src/work_dirs/phase65a/split_lock.json'
    assert sha(split)==cfg['original_split_sha256']
    splitdev={(g['group_id'],g['representative']) for g in load(split)['proposed_rows'] if g['split']=='development_val'}
    assert splitdev=={(g['group_id'],g['product']) for g in dev}
    for g in dev:
        array,valid,_=catalog_image(g)
        actual=plan_indexes(array)
        # Reused E was evaluated with these exact raw patches in Phase117.
        assert actual==g['actual_patch_coordinates'],'old E has different author scheduling'
        p=OLD/'scores'/f"O_{g['product']}.npz";old_cache[g['product']]=sha(p)
        rows.append({'pool':'Catalogue','product_id':g['product'],'group':g,'actual_patch_coordinates':actual,
                     'old_score_sha256':old_cache[g['product']]})
        del array,valid;gc.collect()
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=16*1024**3
        write(ROOT/'PREPARE_PROGRESS.json',{'scenes':len(rows),'model_calls':0,'last':g['product']})
    pix=sum(len(r['actual_patch_coordinates']) for r in rows if r['pool']=='PixBox')
    cat=sum(len(r['actual_patch_coordinates']) for r in rows if r['pool']=='Catalogue')
    bench=sum(len(r['actual_patch_coordinates']) for r in rows if r['product_id'] in cfg['benchmark_products'])
    estimate=2*(pix+cat)+6*bench+33+4  # conservative constructor allowance; actual calls recorded
    write(ROOT/'INPUT_PATCH_PLAN.json',{'rows':rows,'single_PixBox_patch_calls':pix,'single_Catalogue_patch_calls':cat,
          'single_benchmark_patch_calls':bench,'planned_upper_calls':estimate,'constructor_allowance':33,
          'warmup_calls':4,'planned_cap':4000,'recovery_cap':5000,'new_label_semantics_read':False,
          'Catalogue_old_valid_mask_rule_unchanged':True,'no_fit_products_read':True})
    assert estimate<=4000,'fixed two-pool/benchmark plan exceeds budget; no silent expansion'
    (ROOT/'PREPARE_EXIT_CODE').write_text('0\n')
    print({'planned_upper_calls':estimate,'pix':pix,'cat':cat,'benchmark':bench},flush=True)
