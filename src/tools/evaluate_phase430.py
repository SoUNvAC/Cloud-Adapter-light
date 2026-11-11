"""Fixed Catalogue32 evaluation of source-selected Phase430 checkpoints."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

R = Path('/home/scv/shared/phase430_20261102')
OUT = R/'catalogue_best'
OLD = Path('/home/scv/shared/phase400_20261031')
SRC = Path('/home/scv/Cloud-Adapter-light/src')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    sys.path[:0] = [str(R), str(SRC), str(OLD/'code')]
    from report_phase400 import ap_exact, describe, writecsv
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    register_all_modules()
    import cloud_adapter.models
    import cloud_adapter.datasets
    import phase430_model
    spec = importlib.util.spec_from_file_location('catalogue429_loader', SRC/'cloud_adapter/datasets/phase65_catalogue.py')
    module = importlib.util.module_from_spec(spec)
    from mmseg.registry import TRANSFORMS
    if TRANSFORMS.get('LoadPhase65RGB') is None:
        spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.manual_seed(42901)
    torch.backends.cudnn.benchmark = False
    rows = [r for r in json.loads((OLD/'INPUT_LOCK.json').read_text())['target_rows']
            if r['split'] == 'development_val']
    assert len(rows) == len({r['group_id'] for r in rows}) == 32
    for row in rows:
        for key, suffix in [('rgb_path','_rgb.npy'),('mask_path','_mask.npy'),('image_valid_path','_image_valid.npy')]:
            assert sha(row[key]) == row['prepared_sha256'][suffix]
    candidates = list((R/'work_dirs/original_msre').glob('best_mIoU_iter_*.pth'))
    assert len(candidates) == 1
    checkpoints = {'original_msre':candidates[0]}
    save('INPUT_LOCK.json', dict(rows=rows, checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in checkpoints.items()},
        selection='CloudSEN validation best only', resize=[512,512],
        probability='nonnegative Mask2Former semantic scores divided by channel sum; no extra softmax',
        evaluator_sha256=sha(__file__), metric_sha256=sha(OLD/'code/report_phase400.py')))
    pipeline = Compose([dict(type='LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),
                        dict(type='LoadPhase65Mask'),dict(type='PackSegInputs')])
    summaries, groups = [], []
    for variant, checkpoint in checkpoints.items():
        cfg = Config.fromfile(str(R/'phase430_original_msre_40k.py'))
        for name, digest in json.loads((R/'CODE_LOCK.json').read_text()).items():
            assert sha(R/name) == digest
        model = MODELS.build(cfg.model)
        state = torch.load(checkpoint, map_location='cpu')['state_dict']
        model.load_state_dict(state, strict=True)
        del state
        model.cuda().eval()
        directory = OUT/variant
        directory.mkdir()
        pooled = np.zeros((3,3), dtype=np.int64)
        scores, labels, aps = [], [], []
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        for i, row in enumerate(rows):
            sample = pipeline(dict(img_path=row['rgb_path'],seg_map_path=row['mask_path'],
                seg_fields=[],label_map=None,reduce_zero_label=False))
            with torch.inference_mode():
                result = model.test_step(pseudo_collate([sample]))[0]
            truth = sample['data_samples'].gt_sem_seg.data[0].numpy().astype(np.uint8)
            pred = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            assert truth.shape == pred.shape
            assert np.array_equal(truth, np.load(row['mask_path'], allow_pickle=False))
            valid = truth != 255
            assert np.bincount(truth[valid],minlength=3).tolist() == row['label_counts']
            semantic = result.seg_logits.data.detach().float().clamp_min(0)
            probability = semantic/semantic.sum(0,keepdim=True).clamp_min(1e-12)
            assert torch.isfinite(probability).all()
            assert np.array_equal(probability.argmax(0).cpu().numpy(), pred)
            shadow = probability[2].cpu().numpy()
            cm = np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
            # Independent confusion-matrix implementation.
            assert np.array_equal(cm, np.array([[np.count_nonzero(valid & (truth==a) & (pred==b))
                                                for b in range(3)] for a in range(3)]))
            path = directory/(row['product']+'.npz')
            np.savez(path, truth=truth, pred=pred, shadow=shadow)
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['pred'],pred) and np.array_equal(z['shadow'],shadow)
            s, y = shadow[valid].astype(np.float64), truth[valid]==2
            ap = ap_exact(s,y)
            if ap is not None:
                # Independent threshold-group calculation, including tied scores.
                _, inverse = np.unique(s, return_inverse=True)
                positives = np.bincount(inverse, weights=y.astype(np.int64))[::-1]
                totals = np.bincount(inverse)[::-1]
                reference_ap = np.sum(positives * np.cumsum(positives) / np.cumsum(totals)) / y.sum()
                assert abs(ap-reference_ap) < 1e-10
                aps.append(ap)
            scores.append(s); labels.append(y); pooled += cm
            groups.append(dict(variant=variant,product=row['product'],group_id=row['group_id'],
                **describe(cm),shadow_AP=None if ap is None else ap*100,score_sha256=sha(path)))
            save('STATUS.json',dict(variant=variant,completed=i+1,total=32))
        assert len(aps) == 7
        summary = dict(variant=variant,**describe(pooled),macro_shadow_AP=float(np.mean(aps)*100),
            pooled_shadow_AP=ap_exact(np.concatenate(scores),np.concatenate(labels))*100,
            groups=32,AP_groups=7,seconds=time.monotonic()-begin,
            GPU_peak_allocated=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(checkpoint))
        summaries.append(summary)
        save(variant+'/METRICS.json',summary)
        del model, scores, labels
        torch.cuda.empty_cache()
    writecsv(OUT/'SUMMARY.csv',summaries)
    writecsv(OUT/'PER_GROUP.csv',groups)
    save('COMPLETE.json',dict(passed=True,model_calls=32,training_updates=0,
         independent_confusion_and_AP_checks=True,seconds=time.monotonic()-started,summary=summaries))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if OUT.exists():
            (OUT/'FAILURE.txt').write_text(traceback.format_exc())
        raise
