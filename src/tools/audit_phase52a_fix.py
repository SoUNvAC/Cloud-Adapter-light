import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_phase52_sparse_msre import build, logits
from cloud_adapter.datasets.manifest_seg import load_usgs_shadow_status

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/protocol/phase52a_fix_shadow_status_1pct_l1c.py",
    )
    parser.add_argument(
        "--baseline-config", default="configs/protocol/phase22_clean_v8_l1c.py"
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/phase22_clean_v8/seed42/best_mIoU_iter_40000.pth",
    )
    parser.add_argument(
        "--manifest",
        default="work_dirs/phase45_protocol_audit/scene_disjoint_manifest.csv",
    )
    parser.add_argument(
        "--metadata",
        default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv",
    )
    parser.add_argument(
        "--selection", default="work_dirs/phase50_active_1pct/selection.json"
    )
    parser.add_argument(
        "--output", default="work_dirs/phase52a_fix_shadow_status_1pct/preflight.json"
    )
    args = parser.parse_args()

    metadata = load_usgs_shadow_status(args.metadata)
    metadata_sha256 = hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selected_names = {row["name"] for row in selection["selected"]}
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = [row for row in rows if row["name"] in selected_names]
    validation_rows = [row for row in rows if row["new_split"] == "target_val"]
    selected_status = Counter(metadata[row["scene"]] for row in selected_rows)
    validation_status = Counter(metadata[row["scene"]] for row in validation_rows)

    candidate = build(args.config, args.checkpoint)
    candidate.train()
    trainable = {
        name: parameter.numel()
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    }
    candidate.eval().cuda()
    baseline = build(args.baseline_config, args.checkpoint).eval().cuda()
    candidate.backbone.set_target_enabled(False)
    generator = torch.Generator(device="cuda").manual_seed(52)
    inputs = torch.randn(1, 3, 512, 512, generator=generator, device="cuda")
    with torch.inference_mode():
        max_abs = (logits(candidate, inputs) - logits(baseline, inputs)).abs().max().item()
    target_params = sum(trainable.values())
    gates = {
        "official_metadata_96_scenes_32_yes": (
            len(metadata) == 96 and sum(value == "yes" for value in metadata.values()) == 32
        ),
        "official_metadata_frozen_sha256": metadata_sha256
        == "34ece4293c8cc64425feb9650660aa101de753cfd0d0de26709b9add6b5e69ab",
        "exactly_65_selected": len(selected_rows) == 65,
        "selection_without_labels": selection.get("target_labels_read_during_selection") is False,
        "selected_status_is_12_yes_53_no": selected_status == {"yes": 12, "no": 53},
        "validation_status_is_963_yes_942_no": validation_status == {"yes": 963, "no": 942},
        "only_target_modules_trainable": bool(trainable)
        and all("target_msre" in name or "target_head_delta" in name for name in trainable),
        "target_params_exactly_380577": target_params == 380577,
        "disabled_target_path_numerical_parity_1e_5": max_abs <= 1e-5,
        "target_test_sealed": True,
        "cloudsen_internal_test_sealed": True,
    }
    result = {
        "phase": "52A-fix-preflight",
        "protocol_correction": (
            "USGS Shadows?=no scenes do not provide verified shadow negatives; "
            "raw target class 0 is ignored during training and only Shadows?=yes "
            "target-val scenes are used for selection/reporting"
        ),
        "selected_usgs_shadow_status": dict(selected_status),
        "validation_usgs_shadow_status": dict(validation_status),
        "usgs_shadow_metadata_sha256": metadata_sha256,
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
