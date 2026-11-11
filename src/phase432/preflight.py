"""Isolated ten-step real-data training and validation before the formal runs."""
import argparse
import json
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('variant', choices=['baseline', 'fused'])
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    root = Path(args.work_dir)
    assert not root.exists(), root
    register_all_modules()
    cfg = Config.fromfile(str(Path(__file__).with_name('config.py')))
    cfg.model.backbone.variant = args.variant
    cfg.work_dir = str(root)
    cfg.train_cfg.max_iters = 10
    cfg.train_cfg.val_interval = 10
    cfg.default_hooks.logger.interval = 1
    cfg.default_hooks.checkpoint.interval = 10
    # Preserve formal data transformations, optimizer and schedule.
    cfg.val_dataloader.dataset.indices = 8
    runner = Runner.from_cfg(cfg)
    assert len(runner.train_dataloader.dataset) == 8490
    runner.train()
    assert runner.iter == 10
    assert json.loads((root/'UPDATE_CHECK.json').read_text())['frozen_unchanged']
    (root/'COMPLETE.json').write_text(json.dumps(dict(steps=10, validation_samples=8)))


if __name__ == '__main__':
    main()
