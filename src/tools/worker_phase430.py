"""One full 40k run; failure/completion receipts survive SSH disconnects."""
import os
import sys
import json
import hashlib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = Path('/home/scv/Cloud-Adapter-light/src')
os.chdir(SRC)
sys.path[:0] = [str(ROOT), str(SRC)]


def main():
    import torch
    import cloud_adapter
    import cloud_adapter.datasets
    import cloud_adapter.models
    from mmengine.config import Config
    from mmengine.runner import Runner
    cfg = Config.fromfile(str(ROOT / 'phase430_original_msre_40k.py'))
    assert cfg.load_from is None and cfg.model.type == 'EncoderDecoder'
    assert cfg.train_cfg.max_iters == cfg.param_scheduler[-1]['end'] == 40000
    assert cfg.val_dataloader.dataset.data_prefix.img_path == 'img_dir/val'
    assert cfg.train_dataloader.dataset.data_prefix.img_path == 'img_dir/train'
    pretrained = Path(cfg.model.backbone.init_cfg.checkpoint)
    assert pretrained.is_file()
    cfg.dump(str(ROOT / 'RESOLVED_CONFIG.py'))
    runner = Runner.from_cfg(cfg)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train = runner.train_dataloader.dataset
    val = runner.val_dataloader.dataset
    assert len(train) > 0 and len(val) > 0
    train_paths = {x['img_path'] for x in train.data_list}
    val_paths = {x['img_path'] for x in val.data_list}
    # Serialized mmengine datasets may release data_list.
    if not train_paths:
        train_paths = {train.get_data_info(i)['img_path'] for i in range(len(train))}
    if not val_paths:
        val_paths = {val.get_data_info(i)['img_path'] for i in range(len(val))}
    assert train_paths.isdisjoint(val_paths)
    (ROOT / 'PREFLIGHT.json').write_text(json.dumps({
        'train_samples': len(train), 'val_samples': len(val), 'split_disjoint': True,
        'pretrained': str(pretrained.resolve()),
        'pretrained_SHA256': hashlib.sha256(pretrained.read_bytes()).hexdigest(),
        'load_from': None, 'max_iters': 40000, 'batch_size': 2,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
    }, indent=2))
    runner.train()
    assert runner.iter == 40000
    assert (Path(cfg.work_dir) / 'iter_40000.pth').is_file()
    (ROOT / 'COMPLETE.json').write_text(json.dumps({'steps': runner.iter, 'work_dir': cfg.work_dir}))
    del runner
    import gc, subprocess
    gc.collect(); torch.cuda.empty_cache()
    subprocess.run([sys.executable, '-u', str(ROOT/'evaluate_phase430.py')], check=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT / 'FAILURE.txt').write_text(traceback.format_exc())
        (ROOT / 'EXIT_CODE').write_text('1')
        raise
    else:
        (ROOT / 'EXIT_CODE').write_text('0')
