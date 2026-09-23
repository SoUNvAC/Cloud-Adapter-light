import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_phase12_onnx import build_wrapper  # noqa: E402
from phase54_raw_data import load_manifest, read_multispectral_patch  # noqa: E402
from run_phase54_information_audit import (  # noqa: E402
    SOURCE_TO_TARGET,
    cache_path,
    normalized_probabilities,
)
from validate_phase12_onnxruntime import load_rgb_image  # noqa: E402


def load_cached_probabilities(path):
    with np.load(path) as item:
        return item["probabilities"].astype(np.float32), item["auxiliary"].astype(np.float32)


def score_candidates(args, rows):
    wrapper_args = argparse.Namespace(
        config=args.phase52_config, checkpoint=args.phase52_checkpoint,
        precision="fp16", active_block_indices=None,
    )
    wrapper, _ = build_wrapper(wrapper_args)
    candidates = []
    for index, row in enumerate(rows, 1):
        base, auxiliary = load_cached_probabilities(cache_path(args.cache_root, "val", row))
        rgb = torch.from_numpy(load_rgb_image(Path(row["image_path"]), 512)).cuda()
        with torch.inference_mode():
            phase52 = normalized_probabilities(wrapper(rgb))[0].float().cpu().numpy()
        base_shadow = base[3]
        phase52_shadow = phase52[3]
        probability_difference = float(np.abs(base_shadow - phase52_shadow).mean())
        hard_xor = float(np.logical_xor(
            base.argmax(axis=0) == 3, phase52.argmax(axis=0) == 3
        ).mean())
        dark_fraction = float((auxiliary[:3].mean(axis=0) < 0.25).mean())
        sun_elevation = math.degrees(math.asin(float(np.clip(auxiliary[3, 0, 0], -1, 1))))
        candidates.append({
            **row,
            "probability_difference": probability_difference,
            "hard_shadow_xor_fraction": hard_xor,
            "dark_spectral_fraction": dark_fraction,
            "ranking_score": 0.5 * probability_difference + 0.4 * hard_xor + 0.1 * dark_fraction,
            "sun_elevation": sun_elevation,
            "base_prediction": SOURCE_TO_TARGET[base.argmax(axis=0)].astype(np.uint8),
            "phase52_prediction": SOURCE_TO_TARGET[phase52.argmax(axis=0)].astype(np.uint8),
        })
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"label-audit scoring: {index}/{len(rows)}", flush=True)
    return candidates


def assign_solar_strata(candidates):
    scene_elevations = {row["scene"]: row["sun_elevation"] for row in candidates}
    cuts = np.quantile(list(scene_elevations.values()), [1 / 3, 2 / 3])
    for row in candidates:
        row["solar_stratum"] = int(np.searchsorted(cuts, row["sun_elevation"], side="right"))
    return [float(value) for value in cuts]


def select_balanced(candidates):
    selected = []
    for biome in sorted({row["biome"] for row in candidates}):
        pool = sorted(
            (row for row in candidates if row["biome"] == biome),
            key=lambda row: (-row["ranking_score"], row["name"]),
        )
        chosen, scene_counts = [], Counter()
        for row in pool:
            if scene_counts[row["scene"]] >= 9:
                continue
            chosen.append(row)
            scene_counts[row["scene"]] += 1
            if len(chosen) == 18:
                break
        if len(chosen) != 18:
            raise RuntimeError(f"Could not select 18 rows for {biome}")
        selected.extend(chosen)
    for wanted in range(3):
        while sum(row["solar_stratum"] == wanted for row in selected) < 24:
            selected_names = {row["name"] for row in selected}
            counts = Counter(row["solar_stratum"] for row in selected)
            swapped = False
            incoming_rows = sorted(
                (row for row in candidates if row["solar_stratum"] == wanted and row["name"] not in selected_names),
                key=lambda row: (-row["ranking_score"], row["name"]),
            )
            for incoming in incoming_rows:
                same_biome = [row for row in selected if row["biome"] == incoming["biome"]]
                scene_counts = Counter(row["scene"] for row in same_biome)
                if scene_counts[incoming["scene"]] >= 9:
                    continue
                outgoing_rows = [
                    row for row in same_biome
                    if row["solar_stratum"] != wanted and counts[row["solar_stratum"]] > 24
                ]
                if not outgoing_rows:
                    continue
                outgoing = min(outgoing_rows, key=lambda row: (row["ranking_score"], row["name"]))
                selected.remove(outgoing)
                selected.append(incoming)
                swapped = True
                break
            if not swapped:
                raise RuntimeError(f"Cannot satisfy solar stratum {wanted} quota")
    selected.sort(key=lambda row: row["name"])
    return selected


def save_review_package(args, selected, cuts):
    root = Path(args.output_root)
    predictions, templates, figures = (
        root / "predictions", root / "review_records", root / "figures"
    )
    for directory in (predictions, templates, figures):
        directory.mkdir(parents=True, exist_ok=True)
    public_rows = []
    for row in selected:
        stem = Path(row["name"]).stem
        Image.fromarray(row["base_prediction"]).save(predictions / f"{stem}_phase22.png")
        Image.fromarray(row["phase52_prediction"]).save(predictions / f"{stem}_phase52a.png")
        raw, _solar = read_multispectral_patch(args.raw_root, row)
        rgb = np.asarray(Image.open(row["image_path"]))
        label = np.asarray(Image.open(row["mask_path"]))
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB")
        axes[0, 1].imshow(raw[..., 3], cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("NIR B5")
        axes[0, 2].imshow(np.clip(np.stack((raw[..., 5], raw[..., 3], raw[..., 0]), -1), 0, 1))
        axes[0, 2].set_title("SWIR2/NIR/Red")
        axes[1, 0].imshow(label == 1, cmap="gray"); axes[1, 0].set_title("Original shadow")
        axes[1, 1].imshow(row["base_prediction"] == 1, cmap="gray"); axes[1, 1].set_title("Phase 22 shadow")
        axes[1, 2].imshow(row["phase52_prediction"] == 1, cmap="gray"); axes[1, 2].set_title("Phase 52A shadow")
        for axis in axes.flat:
            axis.axis("off")
        figure.suptitle(f"{row['biome']} | sun {row['sun_elevation']:.1f} | score {row['ranking_score']:.4f}")
        figure.tight_layout(); figure.savefig(figures / f"{stem}.png", dpi=120); plt.close(figure)
        template = {
            "name": row["name"],
            "reviewer1": {"id": "", "mask_path": "", "completed_at": ""},
            "reviewer2": {"id": "", "mask_path": "", "completed_at": ""},
            "adjudication": {
                "id": "", "mask_path": "", "completed_at": "",
                "reason_codes": [], "ambiguous_components": 0,
            },
        }
        (templates / f"{stem}.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
        public_rows.append({key: value for key, value in row.items() if not isinstance(value, np.ndarray)})
    with (root / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader(); writer.writerows(public_rows)
    summary = {
        "phase": "54B-selection", "selected_images": len(selected),
        "biome_counts": dict(Counter(row["biome"] for row in selected)),
        "solar_tertile_cuts_degrees": cuts,
        "solar_stratum_counts": dict(Counter(row["solar_stratum"] for row in selected)),
        "review_complete": False, "target_test_read": False,
        "cloudsen_internal_test_read": False,
    }
    (root / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv")
    parser.add_argument("--cache-root", default="work_dirs/phase54_information_audit/cache")
    parser.add_argument("--raw-root", default="/home/scv/shared/data/l8_biome_raw")
    parser.add_argument("--phase52-config", default="configs/protocol/phase52_sparse_msre_1pct_l1c.py")
    parser.add_argument("--phase52-checkpoint", default="work_dirs/phase52_sparse_msre_1pct/seed52/best_mIoU_iter_1000.pth")
    parser.add_argument("--output-root", default="work_dirs/phase54_label_audit")
    args = parser.parse_args()
    rows = load_manifest(args.manifest, "target_val")
    if len(rows) != 1905:
        raise RuntimeError(f"Expected 1,905 target-val rows, found {len(rows)}")
    candidates = score_candidates(args, rows)
    cuts = assign_solar_strata(candidates)
    selected = select_balanced(candidates)
    if len(selected) != 144:
        raise RuntimeError(f"Expected 144 review patches, found {len(selected)}")
    save_review_package(args, selected, cuts)


if __name__ == "__main__":
    main()

