import argparse
import json
from pathlib import Path
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cloud_adapter.models  # noqa: E402,F401


def build(config, checkpoint):
    cfg = Config.fromfile(config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    return model


def logits(model, inputs):
    features = model.extract_feat(inputs)
    return model.decode_head.predict(
        features,
        [dict(ori_shape=(512, 512), img_shape=(512, 512), pad_shape=(512, 512),
              padding_size=[0, 0, 0, 0], flip=False)],
        model.test_cfg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase52_sparse_msre_1pct_l1c.py")
    parser.add_argument("--baseline-config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--output", default="work_dirs/phase52_sparse_msre_1pct/preflight.json")
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    candidate = build(args.config, args.checkpoint)
    candidate.train()
    trainable = {name: parameter.numel() for name, parameter in candidate.named_parameters()
                 if parameter.requires_grad}
    candidate.eval().cuda()
    baseline = build(args.baseline_config, args.checkpoint).eval().cuda()
    candidate.backbone.set_target_enabled(False)
    generator = torch.Generator(device="cuda").manual_seed(52)
    inputs = torch.randn(1, 3, 512, 512, generator=generator, device="cuda")
    with torch.inference_mode():
        difference = (logits(candidate, inputs) - logits(baseline, inputs)).abs()
    max_abs = difference.max().item()
    target_params = sum(trainable.values())
    gates = {
        "exactly_65_selected": selection.get("selected_images") == 65,
        "selection_without_labels": selection.get("target_labels_read_during_selection") is False,
        "only_target_modules_trainable": bool(trainable) and all(
            "target_msre" in name or "target_head_delta" in name for name in trainable
        ),
        "target_params_at_most_500k": target_params <= 500000,
        "disabled_target_path_exact_parity": max_abs == 0.0,
    }
    result = {
        "phase": "52A-preflight",
        "target_trainable_parameters": target_params,
        "trainable_parameter_names": trainable,
        "disabled_path_max_abs": max_abs,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
