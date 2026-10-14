"""Fixed four-run input adaptation. Failures stop; no automatic repeat or fallback."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/scv/shared/phase117_20261014')
DATA = Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
BASE = Path('/home/scv/shared/phase111_ocm_runtime')
LOCK = REPO / 'src/research_results/phase117_20261014/PRE_EXECUTION_LOCK.json'
STATE = {'status': 'preparing', 'calls': 0, 'training_calls': 0, 'gate_calls': 0,
         'evaluation_calls': 0, 'construction_calls': 0}
START = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.writing')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def load(p):
    return json.loads(Path(p).read_text())


def parent_logits(x, torch):
    return torch.stack((x[:, 0], torch.logsumexp(x[:, 1:3], dim=1), x[:, 3]), dim=1)


def arrays(np, iz, mz, group):
    product = group['representative']
    image_bytes = iz.read(f'subscenes/{product}.npy')
    mask_bytes = mz.read(f'masks/{product}.npy')
    raw = np.load(io.BytesIO(image_bytes), allow_pickle=False)
    mask = np.load(io.BytesIO(mask_bytes), allow_pickle=False)
    assert raw.shape == (1022, 1022, 13) and raw.dtype == np.float32
    assert mask.shape == (1022, 1022, 3) and mask.dtype == np.bool_
    band_ok = np.isfinite(raw) & (raw != 0) & (raw != np.float32(65535 / 10000))
    old_observed = np.isfinite(raw) & (raw != 0)
    old_valid = (mask.sum(-1) == 1) & old_observed[..., [3, 2, 1, 7, 11, 12]].all(-1)
    common = old_valid & band_ok[..., [3, 2, 8, 1, 11, 12]].all(-1)
    truth = np.full((1022, 1022), 255, np.uint8)
    truth[common] = mask.argmax(-1)[common]
    six = raw[..., [3, 2, 8, 1, 11, 12]].transpose(2, 0, 1).copy()
    good = band_ok[..., [3, 2, 8, 1, 11, 12]].transpose(2, 0, 1)
    six[~good] = 0
    binding = {'product': product, 'group_id': group['group_id'], 'split': group['split'],
               'image_member_sha256': hashlib.sha256(image_bytes).hexdigest(),
               'mask_member_sha256': hashlib.sha256(mask_bytes).hexdigest(),
               'original_valid': int(old_valid.sum()), 'common_valid': int(common.sum()),
               'support_loss': int((old_valid & ~common).sum()),
               'positive': int((truth == 2).sum()), 'negative': int((truth < 2).sum())}
    return six, truth, binding


def state_hash(models):
    result = []
    for model in models:
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result.append(h.hexdigest())
    return result


def main():
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'PID').write_text(str(os.getpid()))
    cfg = load(LOCK)
    for p, h in cfg['source_sha256'].items():
        assert sha(REPO / p) == h, p
    assert sha(DATA / 'README.pdf') == cfg['official_readme_sha256']
    status = load(DATA / 'download_status.json')
    assert status['status'] == 'all_files_verified'
    split = REPO / 'src/work_dirs/phase65a/split_lock.json'
    assert sha(split) == cfg['original_split_sha256']
    d = load(split)
    groups = [g for g in d['proposed_rows'] if g['split'] in ('fit', 'development_val')]
    fit = [g for g in groups if g['split'] == 'fit']
    dev = [g for g in groups if g['split'] == 'development_val']
    assert len(fit) == 153 and len(dev) == 32
    assert len({g['representative'] for g in groups}) == 185
    original_prepared = DATA / 'phase65a_explore_20261008/manifest.json'
    original_bindings = {r['product']: r for r in load(original_prepared)['rows']}
    assets = load(BASE / 'ASSET_VERIFICATION.json')
    assert assets['status'] == 'all_files_verified'
    for item in assets['files']:
        # Validate the frozen code and weights without opening old evaluation pixels.
        p = Path(item['path'])
        if 'vendor' in p.parts or 'weights' in p.parts:
            assert sha(p) == item['sha256']
    for key in ('HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH',
                'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'TMPDIR'):
        os.environ[key] = str(BASE / 'controlled_caches' / key.lower())
    os.environ.update(HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', GDAL_NUM_THREADS='1')
    sys.path.insert(0, str(BASE / 'assets/vendor'))
    import numpy as np
    import torch
    from omnicloudmask import cloud_mask as cm, model_utils as mu, raster_utils as ru
    from install_phase111_runtime import host_free
    from phase65d_metrics import curve, confusion, measures
    torch.set_num_threads(4)
    assert torch.__version__ == '2.11.0+cu128' and torch.cuda.is_available()
    assert not torch.is_autocast_enabled()
    assert host_free() >= 4000000000
    STATE['host_free_before'] = host_free()
    finished = threading.Event()

    def resource_check():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        gpu = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        STATE.update(rss_peak=max(rss, STATE.get('rss_peak', 0)),
                     gpu_reserved_peak=max(gpu, STATE.get('gpu_reserved_peak', 0)),
                     elapsed_seconds=time.monotonic() - START)
        assert rss <= 16 * 1024**3 and gpu <= 20 * 1024**3
        assert STATE['elapsed_seconds'] <= 21600 and STATE['calls'] <= 9500

    def monitor():
        try:
            while not finished.wait(10):
                resource_check()
                assert host_free() >= 4000000000
                assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) <= 4000000000
                used = int(os.popen('/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').read().strip()) * 1024**2
                STATE['gpu_observed_peak'] = max(STATE.get('gpu_observed_peak', 0), used)
                assert used <= 20 * 1024**3
                dump('RUN_STATUS.json', STATE)
        except Exception:
            STATE.update(status='stopped_resource_limit',failure=traceback.format_exc())
            dump('RUN_STATUS.json', STATE)
            os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=monitor, daemon=True).start()
    iz, mz = zipfile.ZipFile(DATA / 'subscenes.zip'), zipfile.ZipFile(DATA / 'masks.zip')
    bindings = []
    patch_total = 0
    for group in groups:
        six, truth, binding = arrays(np, iz, mz, group)
        old_binding = original_bindings[group['representative']]
        assert binding['image_member_sha256'] == old_binding['member_sha256']['image']
        assert binding['mask_member_sha256'] == old_binding['member_sha256']['mask']
        assert binding['original_valid'] == old_binding['valid_pixels']
        assert binding['common_valid'] > 0
        if group['split'] == 'development_val':
            overlap, size = cm.check_patch_size(six[:3], 0, 1000, 300)
            patches = []
            for index in ru.make_patch_indexes(1022, 1022, size, overlap):
                patch, actual = ru.get_patch(six[:3], index, 0)
                if patch is not None:
                    patches.append(list(actual))
            binding.update(patch_size=size, patch_overlap=overlap, actual_patch_coordinates=patches)
            patch_total += len(patches)
        bindings.append(binding)
    assert sum(b['positive'] > 0 for b in bindings if b['split'] == 'development_val') == 7
    projected = 8000 + 10 * patch_total + 8 + 11
    dump('INPUT_GATE.json', {'status': 'passed' if projected <= 9500 else 'stopped_call_budget',
          'official_cube_order': cfg['band_order'], 'input_indices': [3, 2, 8, 1, 11, 12],
          'base_channels': ['B04', 'B03', 'B8A'], 'extra_channels': ['B02', 'B11', 'B12'],
          'operator': 'official float32 TOA cube, bilinear20m release, not native SAFE reproduction',
          'special_values': [0, float(np.float32(65535 / 10000))],
          'common_support': 'frozen original one-hot/six-band validity AND new six-channel observation validity',
          'development_shadow_groups': 7, 'projected_model_calls': projected, 'bindings': bindings})
    assert projected <= 9500, 'author nodata scheduling exceeds fixed call cap'
    schedules = {}
    for seed in (11701, 11702):
        rng = np.random.default_rng(seed)
        schedules[str(seed)] = [{'fit_index': int(rng.integers(153)), 'top': int(rng.integers(511)),
                                 'left': int(rng.integers(511)), 'horizontal_flip': bool(rng.integers(2))}
                                for _ in range(1000)]
    dump('PAIRED_CROP_SCHEDULE.json', {'fit_order': [g['representative'] for g in fit], 'seeds': schedules})
    model_lock = load(REPO / 'src/research_results/phase111_20261014/PRE_EXECUTION_LOCK.json')
    with (BASE / 'assets/vendor/omnicloudmask/model_download_links.csv').open() as f:
        author_models = list(csv.DictReader(f))
    depth = [0]

    def before(module, args):
        if depth[0] == 0:
            STATE['calls'] += 1
            STATE['construction_calls'] += 1
            dump('RUN_STATUS.json', STATE)
            assert STATE['construction_calls'] <= 11
        depth[0] += 1

    def after(module, args, output):
        depth[0] -= 1

    pre = torch.nn.modules.module.register_module_forward_pre_hook(before)
    post = torch.nn.modules.module.register_module_forward_hook(after, always_call=True)
    models = []
    try:
        for w in model_lock['weights']:
            record = next(r for r in author_models if r['file_name'] == w['google_drive_filename'] and r['version'] == '1')
            models.append(mu.load_model_from_weights(model_name=record['timm_model_name'],
                          weights_path=BASE / 'assets/weights' / w['hf_filename'], model_library='fastai',
                          device=torch.device('cuda'), dtype=torch.float32, in_chans=3, n_out=4,
                          compile_models=False, patch_size=1000, batch_size=1))
    finally:
        pre.remove()
        post.remove()
    for model in models:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    initial_hash = state_hash(models)
    dump('FROZEN_BASE_LOCK.json', {'state_dict_hashes': initial_hash, 'trainable_base_parameters': 0,
                                  'weights': model_lock['weights'], 'eval_always': True})

    def predict(u, kind):
        out = []
        for model in models:
            assert not model.training and not any(m.training for m in model.modules())
            assert not any(p.requires_grad for p in model.parameters())
            STATE['calls'] += 1
            STATE[kind + '_calls'] += 1
            assert STATE['calls'] <= 9500
            out.append(model(u))
        return torch.stack(out).mean(0)

    def new_patch():
        p = torch.nn.Conv2d(6, 3, 1, bias=True, device='cuda', dtype=torch.float32)
        torch.nn.init.zeros_(p.weight)
        torch.nn.init.zeros_(p.bias)
        assert sum(x.numel() for x in p.parameters()) == 21
        return p

    def normalized_crop(record):
        six, truth, binding = arrays(np, iz, mz, fit[record['fit_index']])
        y, x = record['top'], record['left']
        crop = six[:, y:y+512, x:x+512]
        target = truth[y:y+512, x:x+512]
        if record['horizontal_flip']:
            crop, target = crop[:, :, ::-1].copy(), target[:, ::-1].copy()
        assert (target != 255).any(), 'fixed crop has no common labels; no redraw'
        normal = mu.channel_norm(crop, 0)
        assert np.isfinite(normal).all()
        return torch.from_numpy(normal[None]).cuda(), torch.from_numpy(target[None].astype(np.int64)).cuda()

    p = new_patch()
    normal, target = normalized_crop(schedules['11701'][0])
    z, q = normal[:, :3], normal[:, 3:]
    with torch.no_grad():
        baseline = predict(z, 'gate')
        zero = predict(z + .25 * torch.tanh(p(torch.cat((z, q), 1))), 'gate')
        max_diff = float((baseline-zero).abs().max())
    assert max_diff <= 1e-5
    for mode in ('R', 'S'):
        p.zero_grad(set_to_none=True)
        q_gate = z if mode == 'R' else q
        output = predict(z + .25 * torch.tanh(p(torch.cat((z, q_gate), 1))), 'gate')
        loss = torch.nn.functional.cross_entropy(parent_logits(output, torch), target, ignore_index=255)
        loss.backward()
        gradients = torch.cat([v.grad.flatten() for v in p.parameters()])
        assert torch.isfinite(gradients).all() and gradients.abs().max() > 0
        assert all(v.grad is None for m in models for v in m.parameters())
        dump(f'GRADIENT_GATE_{mode}.json', {'finite_nonzero_patch_gradient': True,
             'max_absolute_gradient': float(gradients.abs().max()), 'frozen_base_gradient_none': True})
    assert state_hash(models) == initial_hash
    dump('IMPLEMENTATION_GATE.json', {'status': 'passed', 'zero_patch_max_diff': max_diff,
         'trainable_parameters': 21, 'both_R_S_gradient_passed': True, 'base_state_bitwise_unchanged': True,
         'gate_calls': STATE['gate_calls'], 'construction_calls': STATE['construction_calls']})
    del p, output, baseline, zero, normal, target, loss, z, q, gradients
    torch.cuda.empty_cache()
    trained = {}
    STATE['status'] = 'training'
    for seed in (11701, 11702):
        for mode in ('R', 'S'):
            torch.manual_seed(seed)
            patch = new_patch()
            name = f'{mode}{seed}'
            torch.save(patch.state_dict(), ROOT / f'{name}_initial.pt')
            optimizer = torch.optim.AdamW(patch.parameters(), lr=1e-3, weight_decay=0,
                                         betas=(.9, .999), eps=1e-8)
            losses = []
            for step, record in enumerate(schedules[str(seed)], 1):
                normal, target = normalized_crop(record)
                z = normal[:, :3]
                q = z if mode == 'R' else normal[:, 3:]
                optimizer.zero_grad(set_to_none=True)
                u = z + .25 * torch.tanh(patch(torch.cat((z, q), 1)))
                output = predict(u, 'training')
                loss = torch.nn.functional.cross_entropy(parent_logits(output, torch), target, ignore_index=255)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in patch.parameters())
                assert all(v.grad is None for m in models for v in m.parameters())
                optimizer.step()
                losses.append(float(loss.detach()))
                del normal, target, z, q, u, output, loss
                STATE.update(current_run=name, step=step)
                resource_check()
                if step % 25 == 0:
                    dump('RUN_STATUS.json', STATE)
                    print(name, step, losses[-1], STATE['calls'], flush=True)
            assert state_hash(models) == initial_hash
            torch.save(patch.state_dict(), ROOT / f'{name}_final.pt')
            dump(f'{name}_training.json', {'seed': seed, 'mode': mode, 'steps': 1000, 'losses': losses,
                 'paired_schedule_sha256': sha(ROOT / 'PAIRED_CROP_SCHEDULE.json'),
                 'initial_sha256': sha(ROOT / f'{name}_initial.pt'), 'final_sha256': sha(ROOT / f'{name}_final.pt'),
                 'base_state_bitwise_unchanged': True, 'state_hashes': initial_hash,
                 'trainable_names': ['weight', 'bias'], 'trainable_parameters': 21,
                 'optimizer': 'AdamW lr=.001 wd=0 betas=.9/.999 eps=1e-8 float32 batch1'})
            trained[name] = {k: v.detach().clone() for k, v in patch.state_dict().items()}
            del patch, optimizer
            torch.cuda.empty_cache()
    dump('FOUR_TRAINING_SEAL.json', {'models': {n: sha(ROOT / f'{n}_final.pt') for n in trained},
         'schedule_sha256': sha(ROOT / 'PAIRED_CROP_SCHEDULE.json'), 'code_sha256': sha(Path(__file__)),
         'training_calls': STATE['training_calls'], 'base_state_hashes': initial_hash,
         'no_development_evaluation_before_this_seal': True})
    STATE['status'] = 'evaluating_development'
    policies = ['O', 'R11701', 'S11701', 'R11702', 'S11702']
    result = {n: [] for n in policies}
    (ROOT / 'scores').mkdir()
    for group in dev:
        six, truth, binding = arrays(np, iz, mz, group)
        product = group['representative']
        expected = next(b for b in bindings if b['product'] == product)
        assert binding['image_member_sha256'] == expected['image_member_sha256']
        overlap, size = expected['patch_overlap'], expected['patch_size']
        grad = mu.create_gradient_mask(size, overlap, torch.device('cpu'), torch.float32).numpy()
        prepared = []
        for index in ru.make_patch_indexes(1022, 1022, size, overlap):
            z, actual = ru.get_patch(six[:3], index, 0)
            if z is None:
                continue
            t, b, l, r = actual
            q = mu.channel_norm(six[3:, t:b, l:r], 0)
            prepared.append((actual, z, q))
        assert [list(x[0]) for x in prepared] == expected['actual_patch_coordinates']
        for name in policies:
            patch = None
            if name != 'O':
                patch = new_patch()
                patch.load_state_dict(trained[name], strict=True)
            accumulator = np.zeros((4, 1022, 1022), np.float32)
            coverage = np.zeros((1022, 1022), np.float32)
            with torch.no_grad():
                for (t, b, l, r), z, q in prepared:
                    zt = torch.from_numpy(z[None]).cuda()
                    qt = torch.from_numpy(q[None]).cuda() if name.startswith('S') else zt
                    u = zt if patch is None else zt + .25 * torch.tanh(patch(torch.cat((zt, qt), 1)))
                    output = predict(u, 'evaluation').cpu().numpy()[0]
                    accumulator[:, t:b, l:r] += output * grad
                    coverage[t:b, l:r] += grad
                    del zt, qt, u, output
            valid = truth != 255
            assert (coverage[valid] > 0).all()
            raw = accumulator / np.maximum(coverage[None], np.finfo(np.float32).tiny)
            logits = torch.from_numpy(raw)
            soft = logits.softmax(0).numpy()
            confidence = np.clip(soft[3] + np.float32(.001), .001, .999).astype(np.float32)
            prediction = np.stack((soft[0], soft[1] + soft[2], soft[3])).argmax(0).astype(np.uint8)
            scores = confidence[valid]
            labels = truth[valid]
            cmatrix = confusion(labels, prediction[valid])
            rank = curve(scores, labels == 2)
            result[name].append({'product': product, 'group_id': group['group_id'],
                                 'ap': rank['ap'], **measures(cmatrix)})
            np.savez_compressed(ROOT / 'scores' / f'{name}_{product}.npz', scores=scores, truth=labels,
                                prediction=prediction[valid])
            del patch, accumulator, coverage, raw, logits, soft, confidence, prediction, scores, labels
            torch.cuda.empty_cache()
            resource_check()
        dump('DEVELOPMENT_GROUP_RESULTS.json', result)
        print('evaluated', product, STATE['calls'], flush=True)
    assert state_hash(models) == initial_hash
    dump('FINAL_BASE_UNCHANGED.json', {'bitwise_unchanged': True, 'state_dict_sha256': initial_hash,
         'all_base_gradients_none': all(v.grad is None for m in models for v in m.parameters())})
    STATE.update(status='training_and_development_complete_pending_cpu_acceptance')
    resource_check()
    finished.set()
    dump('RUN_STATUS.json', STATE)
    (ROOT / 'EXIT_CODE').write_text('0\n')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if ROOT.exists():
            (ROOT / 'FAILURE.txt').write_text(traceback.format_exc())
            STATE.update(status='stopped_failure_no_automatic_repeat', failure=traceback.format_exc(),
                         elapsed_seconds=time.monotonic()-START)
            dump('RUN_STATUS.json', STATE)
            (ROOT / 'EXIT_CODE').write_text('1\n')
        raise
