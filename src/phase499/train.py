"""python -m phase499.train {baseline,fused} --work-dir PATH [--resume]"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('variant', choices=['control','texture'])
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise RuntimeError('Use an audited recovery runner: fresh-initialization checks are not resume checks')
    root = Path(args.work_dir)
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError('Nonempty work directory: choose a new path or explicitly resume')
    register_all_modules()
    cfg = Config.fromfile(str(Path(__file__).with_name('config.py')))
    cfg.model.backbone.variant = args.variant
    cfg.work_dir, cfg.resume = str(root), args.resume
    weights = cfg.model.backbone.weights
    if not weights or not Path(weights).is_file():
        raise FileNotFoundError('DINOV3_WEIGHTS must point to the official S+/16 checkpoint')
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'weights_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        'dinov3_commit':subprocess.check_output(['git','-C',cfg.model.backbone.repo,'rev-parse','HEAD'],text=True).strip(),
        'dinov3_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',cfg.model.backbone.repo,'diff'])).hexdigest(),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    if not args.resume:
        (root/'INPUT_LOCK.json').write_text(json.dumps(manifest,indent=2))
    else:
        assert json.loads((root/'INPUT_LOCK.json').read_text()) == manifest, 'Resume inputs/code changed'
    runner = Runner.from_cfg(cfg)
    runner.train()
    assert runner.iter == 40000
    (root/'COMPLETE.json').write_text(json.dumps({'steps':runner.iter,'variant':args.variant}))

if __name__ == '__main__':
    main()
