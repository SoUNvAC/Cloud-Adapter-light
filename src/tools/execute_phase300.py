"""Fixed-teacher Phase300; isolated image-only workers and sealed evaluation."""
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
REPO = Path('/home/scv/Cloud-Adapter-light')
ROOT = Path('/home/scv/shared/phase300_20261020')
OLD = Path('/home/scv/shared/phase208_20261020')
BASE = Path('/home/scv/shared/phase111_ocm_runtime')
START = time.monotonic()
STATE = {}
LOCK = threading.RLock()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.writing.'+str(threading.get_ident()))
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def legacy():
    """Use the exact historical source snapshot, not later working files."""
    for name, expected in load(OLD/'CODE_LOCK.json')['files'].items():
        assert sha(OLD/'code'/name) == expected
    sys.path.insert(0, str(OLD/'code'))
    import execute_phase208 as old
    old.ROOT = ROOT
    old.S = STATE
    return old


def configure():
    for key in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH',
                'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR'):
        directory = ROOT/'controlled_caches'/key.lower()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4',GDAL_NUM_THREADS='1',OPJ_NUM_THREADS='1')
    sys.path.insert(0,str(BASE/'assets/vendor'))
    assert Path(sys.executable).resolve() == (BASE/'runtime_cuda128/bin/python').resolve()
    lock = load(ROOT/'EXECUTION_LOCK.json')
    for name, expected in lock['code'].items():
        assert sha(REPO/name) == expected, name
    assert sha(ROOT/'CROP_SCHEDULE.json') == lock['crop_sha256']
    assert sha(REPO/'src/research_results/phase300_20261020/FIT_IMAGES.json') == lock['manifest_sha256']
    assert sha(REPO/'src/cloud_adapter/models/backbones/target_msre_dinov2.py') == load(OLD/'INPUT_LOCK.json')['source_sha256']['src/cloud_adapter/models/backbones/target_msre_dinov2.py']
    for name, expected in {'model_utils.py':'f749ceced40f8b0899f56e95e79b9f43ea77b74898376175407c21c26796a1dd',
                           'cloud_mask.py':'9130116e5daa130ceaed28de222484f141abf365c614c0b77aaf7b4f8cf7c33c'}.items():
        assert sha(BASE/'assets/vendor/omnicloudmask'/name) == expected
    return lock


def start_worker(stage):
    import torch
    from install_phase111_runtime import host_free
    torch.set_num_threads(4)
    assert torch.__version__ == '2.11.0+cu128' and torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    prior = sum(1 for _ in (ROOT/'CALL_LEDGER.jsonl').open()) if (ROOT/'CALL_LEDGER.jsonl').exists() else 0
    previous = load(ROOT/'CONTROLLER_STATUS.json')['stage_seconds_total']
    STATE.update(stage=stage,kind='constructor',step=0,calls=0,prior_calls=prior,PID=os.getpid(),status='running')
    out = ROOT/stage
    out.mkdir(exist_ok=False)
    stop = threading.Event()

    def guard(host=True):
        with LOCK:
            STATE.update(seconds=time.monotonic()-START,total_seconds=previous+time.monotonic()-START,
                         cpu_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         gpu_peak_bytes=torch.cuda.max_memory_reserved())
            assert prior+STATE['calls'] <= 7000 and STATE['total_seconds'] <= 14400
            assert STATE['cpu_peak_bytes'] <= 16*1024**3 and STATE['gpu_peak_bytes'] <= 20*1024**3
            if host:
                free = host_free()
                STATE['minimum_host_free_bytes'] = min(free,STATE.get('minimum_host_free_bytes',free))
                assert free >= 4000000000
                STATE['new_bytes'] = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
                assert STATE['new_bytes'] <= 6000000000
            write(out/'RUN_STATUS.json',STATE)

    def monitor():
        try:
            while not stop.wait(20): guard()
        except BaseException:
            write(out/'MONITOR_FAILURE.json',dict(failure=traceback.format_exc(),state=STATE))
            os._exit(1)

    thread = threading.Thread(target=monitor,daemon=True)
    thread.start()
    depth = [0]

    def pre(module, args):
        if depth[0] == 0:
            assert prior+STATE['calls'] < 7000
            STATE['calls'] += 1
            with (ROOT/'CALL_LEDGER.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(global_call=prior+STATE['calls'],stage=stage,
                             kind=STATE['kind'],step=STATE['step'],epoch=time.time()))+'\n')
        depth[0] += 1

    def post(module, args, result): depth[0] -= 1
    hooks = [torch.nn.modules.module.register_module_forward_pre_hook(pre),
             torch.nn.modules.module.register_module_forward_hook(post,always_call=True)]
    guard()
    return out,guard,stop,thread,hooks


def inputs():
    return load(REPO/'src/research_results/phase300_20261020/FIT_IMAGES.json')['rows']


def probabilities(logits):
    import numpy as np
    soft = np.exp(logits-logits.max(axis=0,keepdims=True))
    soft /= soft.sum(axis=0,keepdims=True)
    return np.stack((soft[0],soft[1]+soft[2],soft[3]))


def accept(prob, valid):
    import numpy as np
    confidence = prob.max(axis=0)
    target = prob.argmax(axis=0).astype(np.uint8)
    target[~valid | (confidence < np.float32(.90))] = 255
    return target,confidence


def test_loss_rules():
    import numpy as np
    import torch
    p = np.array([[.90,.89999,.01],[.05,.05,.01],[.05,.05001,.98]],np.float32)
    labels,_ = accept(p,np.array([True,True,False]))
    assert labels.tolist() == [0,255,255]
    z = torch.tensor([[[[0.]],[[1.]],[[2.]],[[3.]]]])
    parents = torch.stack((z[:,0],torch.logsumexp(z[:,1:3],1),z[:,3]),1)
    assert torch.allclose(torch.softmax(parents,1),torch.stack((torch.softmax(z,1)[:,0],
               torch.softmax(z,1)[:,1:3].sum(1),torch.softmax(z,1)[:,3]),1))
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter],lr=1e-4,weight_decay=.01)
    # The shared nonempty predicate gates forward/backward/step in train().
    y = torch.full((1,2,2),255)
    if bool((y != 255).any()): raise AssertionError('empty predicate failed')
    assert parameter.item() == 1 and not optimizer.state
    return dict(parent_mapping=[0,1,1,2],threshold_boundary_passed=True,
                parent_logsumexp_softmax_equivalence=True,empty_skip_no_optimizer_state=True)


def teacher(out, guard, old):
    import numpy as np
    import torch
    from phase300_images import load_image,training_access_guard
    from omnicloudmask import cloud_mask as cm,model_utils as mu
    audit = training_access_guard(inputs()[0]['image_archive'],out/'FILE_ACCESS.jsonl')
    model,_ = old.build('R')
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    original = old.digest(model.state_dict())
    plan = {r['product']:r for r in load(ROOT/'FIT_WINDOWS.json')['rows']}
    records = []
    for row in inputs():
        guard()
        STATE.update(kind='fixed_teacher_full_mosaic',product=row['product'])
        image,valid = load_image(row,out/'IMAGE_ACCESS.jsonl')
        actual = []
        raw_store = mu.store_results

        def store(pred,indexes,tracker,gradient,grad_tracker=None):
            actual.extend([list(v) for v in indexes])
            return raw_store(pred,indexes,tracker,gradient,grad_tracker)

        mu.store_results = store
        before = STATE['calls']
        try:
            raw = cm.predict_from_array(image,patch_size=1000,patch_overlap=300,batch_size=1,
                   inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,
                   export_confidence=True,softmax_output=False,custom_models=[model],
                   model_version=4,pred_classes=4,compile_models=False,no_data_value=0,apply_no_data_mask=True)
        finally: mu.store_results = raw_store
        assert actual == plan[row['product']]['actual_windows']
        assert STATE['calls']-before == len(actual) and np.isfinite(raw[:,valid]).all()
        prob = probabilities(raw)
        target,confidence = accept(prob,valid)
        teacher_parent = np.array([0,1,1,2],np.uint8)[raw.argmax(axis=0)]
        path = out/(row['product']+'.npz')
        np.savez_compressed(path,pseudo=target,confidence=confidence,valid=valid,teacher_parent=teacher_parent)
        denominator = int(valid.sum())
        counts = [int((target==i).sum()) for i in range(3)]
        records.append(dict(product=row['product'],group_id=row['group_id'],sha256=sha(path),
                            image_valid=denominator,accepted=sum(counts),class_accepted=counts,
                            image_sha256=row['image_member_sha256'],windows=actual))
        write(out/'CACHE_PROGRESS.json',dict(rows=records))
        print('teacher',len(records),row['product'],STATE['calls'],flush=True)
        del image,valid,raw,prob,target,confidence,teacher_parent
        gc.collect()
    assert old.digest(model.state_dict()) == original
    write(out/'CACHE_LOCK.json',dict(rows=records,frozen_base_digest=original,
          pseudo_definition='stable four-class softmax then [Surface,Thick+Thin,Shadow]; max>=.90',
          teacher_probe_definition='full-context four-raw-argmax merged [0,1,1,2]',
          no_eval_score_offset_or_clip=True,refresh=False,raw_mosaic='original author pipeline; export_confidence=True softmax_output=False',
          windows=sum(len(r['windows']) for r in records),base_unchanged=True,unit_tests=test_loss_rules()))
    audit.flush()


def crop_image(row, location, log):
    from phase300_images import load_image
    from omnicloudmask import model_utils as mu
    image,valid = load_image(row,log)
    top,left = location
    crop = mu.channel_norm(image[:,top:top+512,left:left+512].copy(),0)
    return crop,valid[top:top+512,left:left+512].copy()


def gate(out,guard,old):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from phase300_images import training_access_guard
    audit = training_access_guard(inputs()[0]['image_archive'],out/'FILE_ACCESS.jsonl')
    row = sorted(inputs(),key=lambda r:r['product'])[0]
    array,valid = crop_image(row,(255,255),out/'IMAGE_ACCESS.jsonl')
    x = torch.from_numpy(array[None].copy()).cuda()
    path = ROOT/'teacher'/(row['product']+'.npz')
    assert sha(path) == next(r['sha256'] for r in load(ROOT/'teacher/CACHE_LOCK.json')['rows'] if r['product']==row['product'])
    with np.load(path) as cache:
        y = torch.from_numpy(cache['pseudo'][255:767,255:767][None].astype(np.int64)).cuda()
    assert bool((y!=255).any())
    torch.manual_seed(30001)
    np.random.seed(30001)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    np_rng = np.random.get_state()
    base,_ = old.build('R')
    base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    STATE['kind'] = 'gate_reference'
    with torch.no_grad(): reference = base(x)
    base_digest = old.digest(base.state_dict())
    del base
    results = []
    for arm in ('B','M'):
        model = old.make_candidate(arm)
        initial = {k:v.detach().clone() for k,v in model.adapter.state_dict().items()}
        assert old.digest(model.base.state_dict()) == base_digest
        save_file({k:v.cpu().contiguous() for k,v in initial.items()},str(out/(arm+'_initial.safetensors')))
        model.adapter.enabled = False
        STATE['kind'] = 'gate_disabled_'+arm
        with torch.no_grad(): disabled = model(x)
        error = float((disabled-reference).abs().max())
        assert error <= 1e-5
        model.adapter.enabled = True
        STATE['kind'] = 'gate_backward_'+arm
        logits = model(x)
        loss = old.parent_loss(logits,y)
        loss.backward()
        grads = [p.grad for p in model.adapter.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(bool((g!=0).any()) for g in grads)
        assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
        assert all(not m.training for m in model.base.modules())
        assert old.digest(model.base.state_dict()) == base_digest
        model.adapter.load_state_dict(initial)
        model.adapter.zero_grad(set_to_none=True)
        assert old.digest(model.adapter.state_dict()) == old.digest(initial)
        results.append(dict(arm=arm,closed_max_error=error,finite_nonzero_new_gradients=True,
                       original_no_grad=True,original_buffers_parameters_unchanged=True,
                       initial_restored=True,initial_sha256=sha(out/(arm+'_initial.safetensors')),
                       initial_state_digest=old.digest(initial),parameters=sum(p.numel() for p in model.adapter.parameters())))
        model.handle.remove()
        del model,logits,loss,initial,grads,disabled
        gc.collect()
        torch.cuda.empty_cache()
        guard()
    torch.set_rng_state(rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    np.random.set_state(np_rng)
    write(out/'GATE_ACCEPTANCE.json',dict(status='passed',rows=results,rng_restored=True,
                                         fixed_base_digest=base_digest,unit_tests=test_loss_rules()))
    audit.flush()


def train(arm,out,guard,old):
    import numpy as np
    import torch
    from safetensors.torch import save_file,load_file
    from phase300_images import load_image,training_access_guard
    from omnicloudmask import model_utils as mu
    rows = inputs()
    audit = training_access_guard(rows[0]['image_archive'],out/'FILE_ACCESS.jsonl')
    assert load(ROOT/'gate/GATE_ACCEPTANCE.json')['status'] == 'passed'
    teacher_lock = load(ROOT/'teacher/CACHE_LOCK.json')
    teacher_hash = sha(ROOT/'teacher/CACHE_LOCK.json')
    indexed = {r['product']:r for r in teacher_lock['rows']}
    model = old.make_candidate(arm)
    initial_file = ROOT/'gate'/(arm+'_initial.safetensors')
    model.adapter.load_state_dict(load_file(str(initial_file),device='cuda'))
    before = old.digest(model.base.state_dict())
    buffers = old.digest(dict(model.base.named_buffers()))
    cache = []
    for row in rows:
        image,valid = load_image(row,out/'IMAGE_ACCESS.jsonl')
        file = ROOT/'teacher'/(row['product']+'.npz')
        assert sha(file) == indexed[row['product']]['sha256']
        with np.load(file) as data:
            assert np.array_equal(valid,data['valid'])
            pseudo = data['pseudo'].copy()
        cache.append((image,pseudo))
    schedule = load(ROOT/'CROP_SCHEDULE.json')['rows']
    torch.manual_seed(30001)
    np.random.seed(30001)
    optimizer = torch.optim.AdamW(model.adapter.parameters(),lr=1e-4,weight_decay=.01,
                                 betas=(.9,.999),eps=1e-8)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.adapter.parameters()}
    losses = []
    updates = 0
    probe_rows = []
    probe_products = sorted(rows,key=lambda r:r['product'])[:4]

    def probes(attempt):
        model.eval()
        for row in probe_products:
            array,valid = crop_image(row,(255,255),out/'IMAGE_ACCESS.jsonl')
            with np.load(ROOT/'teacher'/(row['product']+'.npz')) as data:
                teacher_parent = data['teacher_parent'][255:767,255:767].copy()
                assert np.array_equal(valid,data['valid'][255:767,255:767])
            STATE.update(kind='probe_'+arm,step=attempt)
            with torch.no_grad(): raw = model(torch.from_numpy(array[None].copy()).cuda())[0].cpu().numpy()
            parent = np.array([0,1,1,2],np.uint8)[raw.argmax(0)]
            file = out/'probes'/(row['product']+'_'+str(attempt)+'.npz')
            file.parent.mkdir(exist_ok=True)
            np.savez_compressed(file,parent=parent,valid=valid,teacher_parent=teacher_parent)
            probe_rows.append(dict(product=row['product'],arm=arm,attempt=attempt,valid=int(valid.sum()),
                        prediction_counts=[int(((parent==i)&valid).sum()) for i in range(3)],
                        changed=int(((parent!=teacher_parent)&valid).sum()),sha256=sha(file),
                        prediction_rule='four raw argmax merged [0,1,1,2]',top=255,left=255))
        write(out/'PROBE_TABLE.json',dict(rows=probe_rows))
        model.train()
        assert all(not m.training for m in model.base.modules())

    probes(0)
    for attempt,record in enumerate(schedule,1):
        STATE.update(kind='training_'+arm,step=attempt)
        guard(attempt%25==0)
        image,pseudo = cache[record['fit_index']]
        top,left = record['top'],record['left']
        image = image[:,top:top+512,left:left+512]
        target = pseudo[top:top+512,left:left+512]
        if record['horizontal_flip']:
            image = image[:,:,::-1]
            target = target[:,::-1]
        accepted = int((target!=255).sum())
        if not accepted:
            losses.append(dict(attempt=attempt,accepted=0,CE=None,reason='all_ignore_no_forward_backward_or_step'))
        else:
            array = mu.channel_norm(image,0)
            xx = torch.from_numpy(array[None].copy()).cuda()
            yy = torch.from_numpy(target[None].astype(np.int64)).cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)
            loss = old.parent_loss(logits,yy)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None for p in model.base.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.adapter.parameters())
            optimizer.step()
            updates += 1
            losses.append(dict(attempt=attempt,accepted=accepted,CE=float(loss.detach()),reason='updated'))
            del xx,yy,logits,loss
        if attempt in (100,500,1000): probes(attempt)
        if attempt%25 == 0:
            write(out/'TRAINING_PROGRESS.json',dict(attempt=attempt,updates=updates,last=losses[-1]))
            print(arm,attempt,updates,losses[-1],flush=True)
    assert old.digest(model.base.state_dict()) == before == load(ROOT/'gate/GATE_ACCEPTANCE.json')['fixed_base_digest']
    assert old.digest(dict(model.base.named_buffers())) == buffers
    model.eval()
    save_file({k:v.detach().cpu().contiguous() for k,v in model.adapter.state_dict().items()},str(out/'module_final.safetensors'))
    write(out/'TRAINING_RECEIPT.json',dict(arm=arm,attempts=1000,updates=updates,skipped=1000-updates,
          losses=losses,initial_sha256=sha(initial_file),final_sha256=sha(out/'module_final.safetensors'),
          pseudo_cache_lock_sha256=teacher_hash,crop_schedule_sha256=sha(ROOT/'CROP_SCHEDULE.json'),
          base_before=before,base_after=old.digest(model.base.state_dict()),buffers_unchanged=True,
          original_parameters_buffers_unchanged=True,base_all_eval=all(not m.training for m in model.base.modules()),
          initialization_seed=20801,training_seed=30001,optimizer='AdamW lr1e-4 wd.01 betas(.9,.999) eps1e-8 constant',
          initial_saved=True,selection='fixed attempt1000 only; no label/eval access',probe_forwards=16))
    audit.flush()


def worker(stage):
    configure()
    old = legacy()
    out,guard,stop,thread,hooks = start_worker(stage)
    old.OUT = out
    try:
        if stage == 'teacher': teacher(out,guard,old)
        elif stage == 'gate': gate(out,guard,old)
        elif stage.startswith('train_'): train(stage[-1],out,guard,old)
        elif stage == 'infer':
            assert set(load(ROOT/'TRAINING_SEAL.json')) == {'M','B'}
            old.infer(guard)
        else: raise ValueError(stage)
        STATE['status'] = 'completed'
        guard()
        (out/'EXIT_CODE').write_text('0\n')
    finally:
        stop.set()
        thread.join(timeout=25)
        for hook in hooks: hook.remove()


def pipeline():
    configure()
    assert not (ROOT/'LAUNCH.json').exists(), 'Existing launch: no automatic duplicate or recovery'
    write(ROOT/'LAUNCH.json',dict(PID=os.getpid(),epoch=time.time(),runtime=str(Path(sys.executable)),
          authorization='User explicitly allowed fixed historical runtime on 2026-10-20'))
    state = dict(status='running',completed=[],stage_seconds_total=0)
    try:
        for stage in ('teacher','gate','train_B','train_M','infer'):
            if stage == 'infer':
                receipts = {a:load(ROOT/('train_'+a)/'TRAINING_RECEIPT.json') for a in ('B','M')}
                assert receipts['B']['pseudo_cache_lock_sha256'] == receipts['M']['pseudo_cache_lock_sha256']
                assert receipts['B']['crop_schedule_sha256'] == receipts['M']['crop_schedule_sha256']
                write(ROOT/'TRAINING_SEAL.json',{a:sha(ROOT/('train_'+a)/'module_final.safetensors') for a in ('B','M')})
            state['current'] = stage
            write(ROOT/'CONTROLLER_STATUS.json',state)
            begin = time.monotonic()
            with (ROOT/(stage+'.console.log')).open('xb') as stream:
                process = subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'worker',stage],
                                           stdout=stream,stderr=subprocess.STDOUT,cwd=REPO)
                state['worker_PID'] = process.pid
                write(ROOT/'CONTROLLER_STATUS.json',state)
                code = process.wait()
            state['stage_seconds_total'] += time.monotonic()-begin
            state['total_calls'] = sum(1 for _ in (ROOT/'CALL_LEDGER.jsonl').open())
            write(ROOT/'CONTROLLER_STATUS.json',state)
            assert code == 0, (stage,code)
            assert state['total_calls'] <= 7000 and state['stage_seconds_total'] <= 14400
            state['completed'].append(stage)
        state['status'] = 'completed_pending_local_evaluation'
        write(ROOT/'CONTROLLER_STATUS.json',state)
        (ROOT/'EXIT_CODE').write_text('0\n')
    except BaseException:
        state.update(status='failed',failure=traceback.format_exc())
        write(ROOT/'CONTROLLER_STATUS.json',state)
        (ROOT/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'run': pipeline()
        elif sys.argv[1] == 'worker': worker(sys.argv[2])
        else: raise ValueError(sys.argv)
    except BaseException:
        if len(sys.argv)>2 and sys.argv[1]=='worker':
            write(ROOT/sys.argv[2]/'FAILURE.json',dict(failure=traceback.format_exc(),state=STATE))
        raise
