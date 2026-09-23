import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper
from run_phase45_source_only import load_rows
from validate_phase12_onnxruntime import load_rgb_image


def metas(batch, size=512):
    return [dict(ori_shape=(size, size), img_shape=(size, size),
                 pad_shape=(size, size), padding_size=[0, 0, 0, 0], flip=False)
            for _ in range(batch)]


def base_outputs(deployment, model, rgb):
    normalized = deployment.normalize(rgb)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, features = model._base_outputs(normalized, metas(rgb.shape[0]))
        base = model._base_factor_logits(logits)
        base = F.interpolate(base, size=features.shape[-2:], mode="bilinear",
                             align_corners=False)
    return base.float(), features.float()


def factors(base, features, weight, bias):
    return base + F.conv2d(features, weight, bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol/phase47_pixel_factorized_v8_l1c.py")
    parser.add_argument("--checkpoint", default="work_dirs/phase47_pixel_factorized_v8/seed42/best_mIoU_iter_1000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--output", default="work_dirs/phase49_target_factor_adapter/final.pth")
    parser.add_argument("--summary", default="work_dirs/phase49_target_factor_adapter/train_summary.json")
    args = parser.parse_args()

    wrapper_args = argparse.Namespace(config=args.config, checkpoint=args.checkpoint,
                                     precision="fp16", active_block_indices=None)
    deployment, _ = build_wrapper(wrapper_args)
    model = deployment.segmentor
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.factor_residual.weight.requires_grad = True
    model.factor_residual.bias.requires_grad = True
    anchor_weight = model.factor_residual.weight.detach().float().clone()
    anchor_bias = model.factor_residual.bias.detach().float().clone()
    model.factor_residual.float()
    optimizer = torch.optim.Adam(model.factor_residual.parameters(), lr=1e-4)

    rows = load_rows(Path(args.manifest), "target_train")
    if len(rows) != 6502:
        raise RuntimeError(f"Expected 6502 target-train images, found {len(rows)}")
    random.Random(49).shuffle(rows)
    generator = torch.Generator(device="cuda").manual_seed(49)
    totals = dict(consistency=0.0, entropy=0.0, occupancy=0.0, anchor=0.0, loss=0.0)
    for index, row in enumerate(rows, 1):
        rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        brightness = 0.8 + 0.4 * torch.rand((), generator=generator, device="cuda")
        contrast = 0.8 + 0.4 * torch.rand((), generator=generator, device="cuda")
        strong = rgb.flip(-1).float()
        mean = strong.mean(dim=(-2, -1), keepdim=True)
        strong = ((strong - mean) * contrast + mean) * brightness
        strong = strong.clamp(0, 255)
        base_a, feature_a = base_outputs(deployment, model, rgb)
        base_b, feature_b = base_outputs(deployment, model, strong)
        base_b, feature_b = base_b.flip(-1), feature_b.flip(-1)

        student_a = factors(base_a, feature_a, model.factor_residual.weight,
                            model.factor_residual.bias).sigmoid()
        student_b = factors(base_b, feature_b, model.factor_residual.weight,
                            model.factor_residual.bias).sigmoid()
        with torch.no_grad():
            teacher_a = factors(base_a, feature_a, anchor_weight, anchor_bias).sigmoid()
            teacher_b = factors(base_b, feature_b, anchor_weight, anchor_bias).sigmoid()
            teacher_occupancy = 0.5 * (teacher_a.mean((0, 2, 3)) + teacher_b.mean((0, 2, 3)))
        consistency = F.mse_loss(student_a, student_b)
        probability = 0.5 * (student_a + student_b)
        entropy = -(probability * probability.clamp_min(1e-6).log()
                    + (1 - probability) * (1 - probability).clamp_min(1e-6).log()).mean()
        occupancy = F.mse_loss(probability.mean((0, 2, 3)), teacher_occupancy)
        anchor = F.mse_loss(model.factor_residual.weight, anchor_weight) + F.mse_loss(
            model.factor_residual.bias, anchor_bias)
        loss = consistency + 0.01 * entropy + 5.0 * occupancy + anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        values = dict(consistency=consistency, entropy=entropy, occupancy=occupancy,
                      anchor=anchor, loss=loss)
        for name, value in values.items():
            totals[name] += float(value.detach())
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"target adaptation: {index}/{len(rows)} loss={float(loss):.6f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {"phase": 49}}, output)
    summary = {
        "phase": 49, "target_train_images": len(rows), "target_labels_read": False,
        "iterations": len(rows), "learning_rate": 1e-4,
        "loss_weights": {"consistency": 1.0, "entropy": 0.01,
                         "occupancy": 5.0, "source_anchor": 1.0},
        "mean_losses": {name: value / len(rows) for name, value in totals.items()},
        "checkpoint": str(output),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
