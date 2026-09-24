import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit_phase52_sparse_msre import build  # noqa: E402
from cloud_adapter.datasets.manifest_seg import load_usgs_shadow_status  # noqa: E402
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


INJECTION_INDICES = (2, 5, 8, 11)
SOURCE_ORDER_THIN = 2
SOURCE_ORDER_SHADOW = 3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(left, right, eps=1e-12):
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    denominator = left.norm() * right.norm()
    if float(denominator) <= eps:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = labels == 1
    negative = ~positive
    count_positive = int(positive.sum())
    count_negative = int(negative.sum())
    if count_positive == 0 or count_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    begin = 0
    while begin < scores.size:
        end = begin + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + 1 + end)
        begin = end
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - count_positive * (count_positive + 1) / 2)
        / (count_positive * count_negative)
    )


def probe_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    prediction = scores >= 0.0
    positive = labels == 1
    true_positive = int(np.count_nonzero(prediction & positive))
    false_positive = int(np.count_nonzero(prediction & ~positive))
    false_negative = int(np.count_nonzero(~prediction & positive))
    true_negative = int(np.count_nonzero(~prediction & ~positive))
    return {
        "auroc": binary_auc(labels, scores),
        "balanced_accuracy": 0.5 * (
            true_positive / max(true_positive + false_negative, 1)
            + true_negative / max(true_negative + false_positive, 1)
        ),
        "shadow_iou_at_zero_threshold": true_positive
        / max(true_positive + false_positive + false_negative, 1),
        "positive_cells": int(positive.sum()),
        "negative_cells": int((~positive).sum()),
    }


def fit_balanced_ridge(features, labels, ridge=0.01):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise RuntimeError("A probe training set must contain both classes")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    design = np.concatenate(
        (normalized, np.ones((normalized.shape[0], 1), dtype=np.float64)), axis=1
    )
    weights = np.empty(labels.size, dtype=np.float64)
    weights[positive] = 0.5 / positive.sum()
    weights[negative] = 0.5 / negative.sum()
    target = np.where(positive, 1.0, -1.0)
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(
        gram + penalty, design.T @ (weights * target)
    )
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def apply_probe(probe, features):
    features = np.asarray(features, dtype=np.float64)
    normalized = (features - probe["mean"]) / probe["scale"]
    return normalized @ probe["coefficient"][:-1] + probe["coefficient"][-1]


def bootstrap_interval(values, statistic=np.mean, seed=55, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = values[generator.integers(0, values.size, values.size)]
        estimates[index] = statistic(sample)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def retained_basis(gradients, energy=0.90, maximum_rank=8):
    matrix = torch.stack(gradients).double()
    matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    squared = singular.square()
    cumulative = torch.cumsum(squared, dim=0) / squared.sum().clamp_min(1e-12)
    rank = int(torch.searchsorted(cumulative, torch.tensor(energy)).item()) + 1
    rank = max(1, min(rank, maximum_rank, right.shape[0]))
    return right[:rank], singular, rank, matrix


def principal_subspace_metrics(thin_gradients, shadow_gradients):
    thin_basis, thin_singular, thin_rank, thin_matrix = retained_basis(thin_gradients)
    shadow_basis, shadow_singular, shadow_rank, shadow_matrix = retained_basis(
        shadow_gradients
    )
    canonical = torch.linalg.svdvals(thin_basis @ shadow_basis.T).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(canonical)).cpu().numpy()
    projected = shadow_matrix @ thin_basis.T
    shadow_energy_fraction = float(
        projected.square().sum() / shadow_matrix.square().sum().clamp_min(1e-12)
    )
    aggregate_cosine = cosine(thin_matrix.sum(dim=0), shadow_matrix.sum(dim=0))
    return {
        "thin_rank": thin_rank,
        "shadow_rank": shadow_rank,
        "thin_singular_values": [float(value) for value in thin_singular],
        "shadow_singular_values": [float(value) for value in shadow_singular],
        "canonical_cosines": [float(value) for value in canonical],
        "principal_angles_degrees": [float(value) for value in angles],
        "smallest_principal_angle_degrees": float(angles.min()),
        "shadow_gradient_energy_in_thin_subspace": shadow_energy_fraction,
        "aggregate_signed_gradient_cosine": aggregate_cosine,
    }


def read_rows(manifest, metadata_path, selection_path):
    metadata = load_usgs_shadow_status(metadata_path)
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selected = {row["name"] for row in selection["selected"]}
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = [
        row for row in rows
        if row["new_split"] == "target_train"
        and row["name"] in selected
        and metadata[row["scene"]] == "yes"
    ]
    validation = [
        row for row in rows
        if row["new_split"] == "target_val" and metadata[row["scene"]] == "yes"
    ]
    train.sort(key=lambda row: row["name"])
    validation.sort(key=lambda row: row["name"])
    if len(selected) != 65 or len(train) != 12:
        raise RuntimeError(f"Expected 65 selected and 12 Shadows?=yes, got {len(selected)}/{len(train)}")
    if len(validation) != 963 or len({row['scene'] for row in validation}) != 8:
        raise RuntimeError("Expected 963 validation patches from eight Shadows?=yes scenes")
    return train, validation


def load_target_mask(row):
    with Image.open(row["mask_path"]) as image:
        return np.asarray(image, dtype=np.uint8).copy()


def normalized_input(model, row):
    rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
    preprocessor = model.data_preprocessor
    mean = preprocessor.mean.cuda().view(1, 3, 1, 1)
    std = preprocessor.std.cuda().view(1, 3, 1, 1)
    # load_rgb_image already returns model-order RGB BCHW in 0..255.
    return (rgb.float() - mean) / std


def backbone_layers_and_outputs(model, inputs, keep_graph=False):
    backbone = model.backbone
    batch, _, height, width = inputs.shape
    cache = backbone.cloud_adapter.cnn(inputs)
    grid_h, grid_w = height // backbone.patch_size, width // backbone.patch_size
    tokens = backbone.prepare_tokens_with_masks(inputs, None)
    layers = []
    injections = {}
    outputs = []
    adapter_slot = 0
    for index, block in enumerate(backbone.blocks):
        if index not in backbone._active_block_index_set:
            continue
        tokens = block(tokens)
        if index in backbone.adapter_index:
            tokens = backbone.cloud_adapter.forward(
                tokens, adapter_slot, batch_first=True, has_cls_token=True, cache=cache
            )
            adapter_slot += 1
        if hasattr(backbone, "target_msre"):
            tokens = backbone.target_msre(tokens, index, grid_h, grid_w)
            if index in INJECTION_INDICES:
                injections[index] = tokens
        layers.append(tokens[:, 1:])
        if index in backbone.out_indices:
            feature = tokens[:, 1:].permute(0, 2, 1).reshape(
                batch, -1, grid_h, grid_w
            ).contiguous()
            if getattr(backbone, "target_enabled", False):
                feature = feature + backbone.target_head_delta[len(outputs)](feature)
            outputs.append(feature)
    if len(layers) != 12 or len(outputs) != 4:
        raise RuntimeError(f"Unexpected backbone trace: {len(layers)} layers/{len(outputs)} outputs")
    outputs[0] = F.interpolate(outputs[0], scale_factor=4, mode="bilinear", align_corners=False)
    outputs[1] = F.interpolate(outputs[1], scale_factor=2, mode="bilinear", align_corners=False)
    outputs[3] = F.interpolate(outputs[3], scale_factor=0.5, mode="bilinear", align_corners=False)
    if not keep_graph:
        layers = [value.detach() for value in layers]
    return layers, outputs, injections


def semantic_probabilities(model, features):
    class_scores, mask_scores = model.decode_head(features, None)
    class_probability = F.softmax(class_scores[-1], dim=-1)[..., :-1]
    mask_probability = mask_scores[-1].sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", class_probability, mask_probability)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)


def reduced_labels(mask, size, source_order=False):
    labels = torch.from_numpy(mask.astype(np.int64)).cuda()[None, None].float()
    labels = F.interpolate(labels, size=size, mode="nearest")[0, 0].long()
    if source_order:
        mapping = torch.tensor([0, 3, 2, 1], device=labels.device)
        labels = mapping[labels]
    return labels


def run_gradient_audits(model, rows):
    model.eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_names, trainable_parameters = [], []
    for name, parameter in model.named_parameters():
        if "target_msre" in name or "target_head_delta" in name:
            parameter.requires_grad_(True)
            trainable_names.append(name)
            trainable_parameters.append(parameter)
    if sum(parameter.numel() for parameter in trainable_parameters) != 380577:
        raise RuntimeError("Phase 55 requires exactly 380,577 target parameters")
    activation_cosines = {index: [] for index in INJECTION_INDICES}
    pooled_thin_activation = {index: None for index in INJECTION_INDICES}
    pooled_shadow_activation = {index: None for index in INJECTION_INDICES}
    token_cosines = {index: [] for index in INJECTION_INDICES}
    thin_parameter_gradients = []
    shadow_parameter_gradients = []
    contributing = []
    token_parameter_index = next(
        index for index, parameter in enumerate(trainable_parameters)
        if parameter is model.backbone.target_msre.learnable_tokens
    )
    for row in rows:
        model.zero_grad(set_to_none=True)
        inputs = normalized_input(model, row)
        layers, outputs, injections = backbone_layers_and_outputs(model, inputs, keep_graph=True)
        probabilities = semantic_probabilities(model, outputs)
        labels = reduced_labels(load_target_mask(row), probabilities.shape[-2:], source_order=True)
        thin = labels == SOURCE_ORDER_THIN
        shadow = labels == SOURCE_ORDER_SHADOW
        if not thin.any() or not shadow.any():
            continue
        thin_loss = -probabilities[0, SOURCE_ORDER_THIN][thin].clamp_min(1e-8).log().mean()
        shadow_loss = -probabilities[0, SOURCE_ORDER_SHADOW][shadow].clamp_min(1e-8).log().mean()
        gradient_targets = [injections[index] for index in INJECTION_INDICES] + trainable_parameters
        thin_gradients = torch.autograd.grad(
            thin_loss, gradient_targets, retain_graph=True, allow_unused=True
        )
        shadow_gradients = torch.autograd.grad(
            shadow_loss, gradient_targets, retain_graph=False, allow_unused=True
        )
        thin_activation = thin_gradients[:4]
        shadow_activation = shadow_gradients[:4]
        thin_parameters = thin_gradients[4:]
        shadow_parameters = shadow_gradients[4:]
        if any(value is None for value in thin_parameters + shadow_parameters):
            raise RuntimeError("A target parameter is disconnected from a diagnostic loss")
        for slot, block_index in enumerate(INJECTION_INDICES):
            activation_cosines[block_index].append(
                cosine(thin_activation[slot], shadow_activation[slot])
            )
            thin_unit = thin_activation[slot].detach().float().reshape(-1).cpu()
            shadow_unit = shadow_activation[slot].detach().float().reshape(-1).cpu()
            thin_unit = thin_unit / thin_unit.norm().clamp_min(1e-12)
            shadow_unit = shadow_unit / shadow_unit.norm().clamp_min(1e-12)
            if pooled_thin_activation[block_index] is None:
                pooled_thin_activation[block_index] = thin_unit
                pooled_shadow_activation[block_index] = shadow_unit
            else:
                pooled_thin_activation[block_index] += thin_unit
                pooled_shadow_activation[block_index] += shadow_unit
        thin_tokens = thin_parameters[token_parameter_index]
        shadow_tokens = shadow_parameters[token_parameter_index]
        for slot, block_index in enumerate(INJECTION_INDICES):
            token_cosines[block_index].append(cosine(thin_tokens[slot], shadow_tokens[slot]))
        thin_parameter_gradients.append(
            torch.cat([value.detach().float().reshape(-1).cpu() for value in thin_parameters])
        )
        shadow_parameter_gradients.append(
            torch.cat([value.detach().float().reshape(-1).cpu() for value in shadow_parameters])
        )
        contributing.append(row["name"])
        print(f"gradient audit: {len(contributing)} valid patches", flush=True)
        del layers, outputs, injections, probabilities, thin_gradients, shadow_gradients
        torch.cuda.empty_cache()
    if len(contributing) < 4:
        raise RuntimeError(f"Too few patches containing both classes: {len(contributing)}")

    def summarize(values):
        array = np.asarray(values, dtype=np.float64)
        return {
            "per_patch": [float(value) for value in array],
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "bootstrap_mean_95pct": bootstrap_interval(array),
        }

    activation = {str(index): summarize(activation_cosines[index]) for index in INJECTION_INDICES}
    token = {str(index): summarize(token_cosines[index]) for index in INJECTION_INDICES}
    pooled_activation = {}
    for index in INJECTION_INDICES:
        pooled_activation[str(index)] = cosine(
            pooled_thin_activation[index], pooled_shadow_activation[index]
        )
    conflict_gate = (
        sum(value < -0.10 for value in pooled_activation.values()) >= 2
        and float(np.mean(list(pooled_activation.values()))) < 0.0
    )
    subspace = principal_subspace_metrics(
        thin_parameter_gradients, shadow_parameter_gradients
    )
    overlap_gate = (
        subspace["smallest_principal_angle_degrees"] <= 75.0
        and subspace["shadow_gradient_energy_in_thin_subspace"] >= 0.10
        and subspace["aggregate_signed_gradient_cosine"] < 0.0
    )
    return {
        "valid_patches": len(contributing),
        "patch_names": contributing,
        "post_msre_activation_gradient_cosine": activation,
        "layer_token_gradient_cosine": token,
        "pooled_post_msre_cosine": pooled_activation,
        "mean_pooled_post_msre_cosine": float(np.mean(list(pooled_activation.values()))),
        "gradient_conflict_gate": bool(conflict_gate),
    }, {**subspace, "overlap_gate": bool(overlap_gate)}


def collect_probe_training(model, rows):
    layer_features = [[] for _ in range(12)]
    labels = []
    for index, row in enumerate(rows, 1):
        inputs = normalized_input(model, row)
        with torch.inference_mode():
            layers, _, _ = backbone_layers_and_outputs(model, inputs)
        grid = int(round(math.sqrt(layers[0].shape[1])))
        mask = reduced_labels(load_target_mask(row), (grid, grid), source_order=False)
        labels.append((mask == 1).cpu().numpy().reshape(-1))
        for layer_index, value in enumerate(layers):
            layer_features[layer_index].append(value[0].float().cpu().numpy())
        print(f"probe train features: {index}/{len(rows)}", flush=True)
    labels = np.concatenate(labels)
    features = [np.concatenate(values, axis=0) for values in layer_features]
    probes = [fit_balanced_ridge(value, labels, ridge=0.01) for value in features]
    return probes, labels


def evaluate_probes(model, rows, probes):
    pooled_labels = []
    pooled_scores = [[] for _ in range(12)]
    by_scene = {}
    for index, row in enumerate(rows, 1):
        inputs = normalized_input(model, row)
        with torch.inference_mode():
            layers, _, _ = backbone_layers_and_outputs(model, inputs)
        grid = int(round(math.sqrt(layers[0].shape[1])))
        labels = (reduced_labels(
            load_target_mask(row), (grid, grid), source_order=False
        ) == 1).cpu().numpy().reshape(-1)
        pooled_labels.append(labels)
        scene = by_scene.setdefault(
            row["scene"], {"labels": [], "scores": [[] for _ in range(12)]}
        )
        scene["labels"].append(labels)
        for layer_index, value in enumerate(layers):
            scores = apply_probe(probes[layer_index], value[0].float().cpu().numpy())
            pooled_scores[layer_index].append(scores)
            scene["scores"][layer_index].append(scores)
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"probe validation: {index}/{len(rows)}", flush=True)
    labels = np.concatenate(pooled_labels)
    pooled = [
        probe_metrics(labels, np.concatenate(pooled_scores[index]))
        for index in range(12)
    ]
    scene_metrics = {}
    for scene, values in sorted(by_scene.items()):
        scene_labels = np.concatenate(values["labels"])
        scene_metrics[scene] = [
            probe_metrics(scene_labels, np.concatenate(values["scores"][index]))
            for index in range(12)
        ]
    return pooled, scene_metrics


def run_probe_model(config, checkpoint, train_rows, validation_rows):
    model = build(config, checkpoint).eval().cuda()
    probes, train_labels = collect_probe_training(model, train_rows)
    pooled, scenes = evaluate_probes(model, validation_rows, probes)
    result = {
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "train_feature_cells": int(train_labels.size),
        "train_shadow_cells": int(train_labels.sum()),
        "layers": {str(index): pooled[index] for index in range(12)},
        "per_scene": {
            scene: {str(index): metrics[index] for index in range(12)}
            for scene, metrics in scenes.items()
        },
    }
    del model
    torch.cuda.empty_cache()
    return result


def probe_comparison(source, candidate):
    layers = {}
    for layer in range(12):
        source_auc = source["layers"][str(layer)]["auroc"]
        candidate_auc = candidate["layers"][str(layer)]["auroc"]
        scene_deltas = []
        for scene in sorted(source["per_scene"]):
            left = source["per_scene"][scene][str(layer)]["auroc"]
            right = candidate["per_scene"][scene][str(layer)]["auroc"]
            if math.isfinite(left) and math.isfinite(right):
                scene_deltas.append(right - left)
        layers[str(layer)] = {
            "source_auroc": source_auc,
            "candidate_auroc": candidate_auc,
            "candidate_minus_source_auroc": candidate_auc - source_auc,
            "per_scene_auroc_deltas": scene_deltas,
            "scene_bootstrap_delta_95pct": bootstrap_interval(scene_deltas),
        }
    injection_drop = any(
        layers[str(index)]["candidate_minus_source_auroc"] <= -0.05
        for index in INJECTION_INDICES
    )
    final_not_recovered = layers["11"]["candidate_minus_source_auroc"] < -0.02
    return {
        "layers": layers,
        "representation_loss_gate": bool(injection_drop and final_not_recovered),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", default="configs/protocol/phase22_clean_v8_l1c.py")
    parser.add_argument("--candidate-config", default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py")
    parser.add_argument("--source-checkpoint", default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth")
    parser.add_argument("--candidate-checkpoint", default="work_dirs/phase52a_fix_shadow_status_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--metadata", default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv")
    parser.add_argument("--selection", default="work_dirs/phase50_active_1pct/selection.json")
    parser.add_argument("--output", default="work_dirs/phase55_thin_shadow_mechanism/summary.json")
    args = parser.parse_args()

    torch.manual_seed(55)
    np.random.seed(55)
    train_rows, validation_rows = read_rows(args.manifest, args.metadata, args.selection)
    candidate = build(args.candidate_config, args.candidate_checkpoint).eval().cuda()
    gradients, subspace = run_gradient_audits(candidate, train_rows)
    del candidate
    torch.cuda.empty_cache()

    source_probe = run_probe_model(
        args.source_config, args.source_checkpoint, train_rows, validation_rows
    )
    candidate_probe = run_probe_model(
        args.candidate_config, args.candidate_checkpoint, train_rows, validation_rows
    )
    comparison = probe_comparison(source_probe, candidate_probe)
    gates = {
        "gradient_conflict": gradients["gradient_conflict_gate"],
        "layerwise_shadow_information_loss": comparison["representation_loss_gate"],
        "thin_shadow_subspace_overlap": subspace["overlap_gate"],
        "expected_train_shadow_status_yes_patches": len(train_rows) == 12,
        "expected_corrected_target_val": len(validation_rows) == 963,
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    mechanism_supported = all(gates.values())
    result = {
        "phase": "55-thin-shadow-asymmetric-negative-transfer-audit",
        "protocol": {
            "injection_indices": list(INJECTION_INDICES),
            "ridge_lambda": 0.01,
            "subspace_energy": 0.90,
            "subspace_max_rank": 8,
            "bootstrap_seed": 55,
            "train_patches": len(train_rows),
            "validation_patches": len(validation_rows),
            "validation_scenes": len({row["scene"] for row in validation_rows}),
            "metadata_sha256": sha256(args.metadata),
        },
        "gradient_geometry": gradients,
        "source_only_layerwise_shadow_probe": source_probe,
        "phase52a_fix_layerwise_shadow_probe": candidate_probe,
        "probe_comparison": comparison,
        "thin_shadow_parameter_subspaces": subspace,
        "gates": gates,
        "passed": mechanism_supported,
        "decision": (
            "thin_shadow_conflict_supported_design_decoupling"
            if mechanism_supported
            else "stop_thin_shadow_conflict_hypothesis"
        ),
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not mechanism_supported:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
