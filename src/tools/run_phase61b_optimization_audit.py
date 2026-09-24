"""Phase 61B: controlled optimization/exposure audit on frozen Phase 60 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from audit_phase56_readout import metrics  # noqa: E402
from phase56_protocol import sha256  # noqa: E402


def array(root, name):
    return np.load(Path(root) / f"{name}.npy", mmap_mode="r")


def stable_seed(name, seed):
    token = hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8]
    return int.from_bytes(token, "little") % (2**31 - 1)


def evaluate(linear, features, labels, chunk=32768):
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), chunk):
            parts.append(linear(features[start:start + chunk]).float().cpu().numpy())
    return metrics(np.concatenate(parts), labels.cpu().numpy())


def full_loss(linear, features, labels, weights, chunk=32768):
    total, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(features), chunk):
            x, y = features[start:start + chunk], labels[start:start + chunk]
            loss = F.cross_entropy(linear(x), y, weight=weights, reduction="sum")
            total += float(loss)
            count += len(x)
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="work_dirs/phase60/phase60b_selections.json")
    parser.add_argument("--train-cache", default="work_dirs/phase60/phase60b_feature_cache/adapted/target_train_union")
    parser.add_argument("--validation-cache", default="work_dirs/phase56_readout/cache/adapted/target_val")
    parser.add_argument("--output", default="work_dirs/phase61/phase61b_optimization.json")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--snapshot-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    train_meta = json.loads((Path(args.train_cache) / "metadata.json").read_text(encoding="utf-8"))
    val_meta = json.loads((Path(args.validation_cache) / "metadata.json").read_text(encoding="utf-8"))
    if train_meta["sampling_seed"] != val_meta["sampling_seed"]:
        raise RuntimeError("Phase 61B requires identical train/validation sampling seeds")
    names = train_meta["image_ids"]
    name_to_image = {name: index for index, name in enumerate(names)}
    train_features = torch.from_numpy(np.asarray(array(args.train_cache, "pixel_decoder_mask"))).cuda().float()
    train_labels = torch.from_numpy(np.asarray(array(args.train_cache, "label"), dtype=np.int64)).cuda()
    image_index = np.asarray(array(args.train_cache, "image_index"), dtype=np.int64)
    val_features = torch.from_numpy(np.asarray(array(args.validation_cache, "pixel_decoder_mask"))).cuda().float()
    val_labels = torch.from_numpy(np.asarray(array(args.validation_cache, "label"), dtype=np.int64)).cuda()
    per_image = {index: np.flatnonzero(image_index == index) for index in range(len(names))}

    requested = []
    for method, descriptor in selection["selections"].items():
        for budget, chosen in descriptor.get("budgets", {}).items():
            requested.append((f"{method}@{budget}", method, int(budget), chosen, descriptor["oracle_upper_bound_only"]))
        if method == "phase50_exact_nominal65":
            requested.append((method, method, 65, descriptor["names"], False))

    results = {}
    for run_number, (key, method, requested_budget, chosen_names, oracle) in enumerate(requested, 1):
        chosen_images = [name_to_image[name] for name in chosen_names]
        selected_rows = np.concatenate([per_image[index] for index in chosen_images])
        x_selected = train_features[torch.from_numpy(selected_rows).cuda()]
        y_selected = train_labels[torch.from_numpy(selected_rows).cuda()]
        class_counts = torch.bincount(y_selected, minlength=4).float()
        if torch.any(class_counts == 0):
            results[key] = {"status": "not_fittable_missing_class", "class_counts": class_counts.cpu().tolist()}
            continue
        class_weights = class_counts.sum() / (4.0 * class_counts)
        run_seed = stable_seed(key, args.seed)
        torch.manual_seed(run_seed)
        generator = np.random.default_rng(run_seed)
        linear = torch.nn.Linear(train_features.shape[1], 4).cuda()
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        exposure = {index: 0 for index in chosen_images}
        losses = [{"step": 0, "full_train_loss": full_loss(linear, x_selected, y_selected, class_weights)}]
        for step in range(1, args.steps + 1):
            batch_images = generator.choice(
                chosen_images, size=args.image_batch_size,
                replace=len(chosen_images) < args.image_batch_size,
            )
            batch_rows = np.concatenate([per_image[int(index)] for index in batch_images])
            for index in batch_images:
                exposure[int(index)] += 1
            batch_index = torch.from_numpy(batch_rows).cuda()
            logits = linear(train_features[batch_index])
            loss = F.cross_entropy(logits, train_labels[batch_index], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % args.snapshot_interval == 0 or step == args.steps:
                losses.append({"step": step, "full_train_loss": full_loss(linear, x_selected, y_selected, class_weights)})

        exposure_values = np.asarray(list(exposure.values()), dtype=np.float64)
        start_loss, end_loss = losses[0]["full_train_loss"], losses[-1]["full_train_loss"]
        last_change = (
            abs(losses[-1]["full_train_loss"] - losses[-2]["full_train_loss"])
            / max(abs(losses[-2]["full_train_loss"]), 1e-12)
        )
        results[key] = {
            "status": "complete", "method": method,
            "oracle_upper_bound_only": oracle,
            "requested_budget": requested_budget, "actual_images": len(chosen_names),
            "optimizer": "AdamW", "actual_optimizer_steps": args.steps,
            "image_batch_size": args.image_batch_size,
            "average_image_exposures": float(exposure_values.mean()),
            "minimum_image_exposures": int(exposure_values.min()),
            "maximum_image_exposures": int(exposure_values.max()),
            "expected_average_exposures": args.steps * args.image_batch_size / len(chosen_names),
            "sampled_pixels": int(len(selected_rows)),
            "effective_pixel_sampling_ratio": {"source": 0.0, "target": 1.0},
            "loss_trace": losses,
            "loss_reduction_fraction": (start_loss - end_loss) / max(start_loss, 1e-12),
            "last_interval_relative_change": last_change,
            "convergence_criterion": "loss decreases and final 500-step relative change <= 1%",
            "loss_converged": end_loss < start_loss and last_change <= 0.01,
            "train_sampled_metrics": evaluate(linear, x_selected, y_selected),
            "validation_sampled_metrics": evaluate(linear, val_features, val_labels),
        }
        print(f"61B {run_number}/{len(requested)} {key}: loss {start_loss:.5f}->{end_loss:.5f}", flush=True)

    random65 = results.get("random@65", {})
    label_aware65 = {
        key: results.get(key, {}) for key in (
            "phase50_policy_rebased@65", "entropy@65", "uncertainty_plus_diversity@65"
        )
    }
    random_weak = random65.get("validation_sampled_metrics", {}).get("weak_harmonic_mean")
    stable_advantage = None
    if random_weak is not None:
        stable_advantage = all(
            random_weak > value.get("validation_sampled_metrics", {}).get("weak_harmonic_mean", float("inf"))
            for value in label_aware65.values()
        )
    summary = {
        "phase": "61B",
        "protocol": {
            "representation": "frozen Phase52 pixel_decoder_mask",
            "optimizer_steps_per_run": args.steps,
            "image_batch_size": args.image_batch_size,
            "sampling": "uniform images with replacement across steps; all cached capped pixels from each exposed image",
            "loss": "class-balanced cross entropy",
            "train_cache_sha256": sha256(Path(args.train_cache) / "metadata.json"),
            "validation_cache_sha256": sha256(Path(args.validation_cache) / "metadata.json"),
            "target_test_evaluated": False,
            "cloudsen_internal_test_evaluated": False,
        },
        "results": results,
        "diagnostics": {
            "random65_strictly_better_than_all_label_aware65_on_sampled_weak_harmonic": stable_advantage,
            "note": "A single deterministic run is not enough to claim statistical stability; repeat seeds would be required for that wording.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(results), "diagnostics": summary["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
