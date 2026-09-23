import argparse
import json
from pathlib import Path
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmseg.registry import DATASETS, DATA_SAMPLERS, MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cloud_adapter.datasets  # noqa: E402,F401
import cloud_adapter.models  # noqa: E402,F401


def build(config, checkpoint):
    cfg = Config.fromfile(config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase52b_shadow_residual_1pct_l1c.py")
    parser.add_argument("--phase52-config", default="configs/protocol/phase52_sparse_msre_1pct_l1c.py")
    parser.add_argument("--phase52-checkpoint", default="work_dirs/phase52_sparse_msre_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--output", default="work_dirs/phase52b_shadow_residual_1pct/preflight.json")
    args = parser.parse_args()

    candidate, cfg = build(args.config, args.phase52_checkpoint)
    candidate.train()
    trainable = {
        name: parameter.numel() for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    }
    trainable_count = sum(trainable.values())
    candidate.eval().cuda()
    generator = torch.Generator(device="cuda").manual_seed(52)
    raw = torch.randn(1, 3, 512, 512, generator=generator, device="cuda") * 40 + 128
    normalized = (raw - candidate.input_mean) / candidate.input_std
    meta = [dict(ori_shape=(512, 512), img_shape=(512, 512), pad_shape=(512, 512),
                 padding_size=[0, 0, 0, 0], flip=False)]
    with torch.inference_mode():
        # Compare before/after the zero residual inside one frozen base forward.
        # Two independent CUDA deformable-attention forwards are not bitwise
        # deterministic and would test kernel noise rather than the algebraic
        # identity of the residual construction.
        base_features, base_probability = candidate._frozen_base(normalized, meta)
        candidate_logits, _ = candidate._adapt(
            normalized, base_features, base_probability
        )
        candidate_probability = candidate_logits.softmax(dim=1)
    max_abs = (base_probability - candidate_probability).abs().max().item()
    prediction_equal = torch.equal(
        base_probability.argmax(dim=1), candidate_probability.argmax(dim=1)
    )

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    sampler_cfg = dict(cfg.train_dataloader.sampler)
    sampler = DATA_SAMPLERS.build(sampler_cfg, default_args=dict(dataset=dataset))
    iterator = iter(sampler)
    diagnosis = json.loads(
        Path(sampler_cfg["diagnosis_path"]).read_text(encoding="utf-8")
    )
    shadow_names = set(diagnosis["selected_1pct_labels"]["shadow_image_names"])
    guaranteed = True
    for _ in range(25):
        indices = [next(iterator) for _ in range(4)]
        names = {Path(dataset.get_data_info(index)["img_path"]).name for index in indices}
        guaranteed &= bool(names & shadow_names)

    gates = {
        "only_shadow_residual_trainable": bool(trainable) and all(
            name.startswith("shadow_residual.") for name in trainable
        ),
        "new_parameters_at_most_100k": trainable_count <= 100000,
        "expected_new_parameter_count_12833": trainable_count == 12833,
        "zero_init_probability_parity_1e_5": max_abs <= 1e-5,
        "zero_init_prediction_exact": prediction_equal,
        "one_shadow_patch_per_batch": guaranteed,
        "phase52_checkpoint_exists": Path(args.phase52_checkpoint).is_file(),
    }
    result = {
        "phase": "52B-preflight",
        "new_trainable_parameters": trainable_count,
        "trainable_parameter_names": trainable,
        "zero_init_probability_max_abs": max_abs,
        "checked_sampler_batches": 25,
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
