import argparse
import hashlib
import json
from pathlib import Path
import sys

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cloud_adapter.models  # noqa: E402,F401


METHOD_CONFIGS = {
    "frozen": "configs/protocol/phase29_frozen_v12_l1c.py",
    "rein": "configs/protocol/phase29_rein_v12_l1c.py",
    "cloud_adapter": "configs/protocol/phase29_cloud_adapter_v12_l1c.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Phase 29 decoder, schedule, split, and PEFT budgets."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-method-param-ratio", type=float, default=2.0)
    return parser.parse_args()


def canonical(value):
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def digest(value):
    payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_parameters(model):
    groups = {
        "deployment": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "backbone_trainable": sum(
            parameter.numel()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ),
        "decode_head_trainable": sum(
            parameter.numel()
            for parameter in model.decode_head.parameters()
            if parameter.requires_grad
        ),
    }
    groups["other_trainable"] = (
        groups["trainable"]
        - groups["backbone_trainable"]
        - groups["decode_head_trainable"]
    )
    return groups


def main():
    args = parse_args()
    if args.max_method_param_ratio < 1.0:
        raise ValueError("max-method-param-ratio must be at least 1")
    rows = {}
    for method, relative_config in METHOD_CONFIGS.items():
        config_path = REPO_ROOT / relative_config
        cfg = Config.fromfile(config_path)
        init_default_scope(cfg.get("default_scope", "mmseg"))
        model = MODELS.build(cfg.model)
        model.train()
        val_prefix = cfg.val_dataloader.dataset.data_prefix
        test_prefix = cfg.test_dataloader.dataset.data_prefix
        schedule_contract = {
            "optim_wrapper": cfg.optim_wrapper,
            "param_scheduler": cfg.param_scheduler,
            "train_cfg": cfg.train_cfg,
            "train_batch_size": cfg.train_dataloader.batch_size,
            "val_batch_size": cfg.val_dataloader.batch_size,
        }
        rows[method] = {
            "config": relative_config,
            "model_type": cfg.model.type,
            "parameters": count_parameters(model),
            "decode_head_sha256": digest(cfg.model.decode_head),
            "schedule_sha256": digest(schedule_contract),
            "load_from": cfg.get("load_from"),
            "val_prefix": canonical(val_prefix),
            "test_prefix": canonical(test_prefix),
            "max_iters": cfg.train_cfg.max_iters,
        }
        print(
            f"{method}: trainable={rows[method]['parameters']['trainable']:,}, "
            f"method={rows[method]['parameters']['backbone_trainable']:,}, "
            f"head={rows[method]['parameters']['decode_head_trainable']:,}"
        )

    decoder_hashes = {row["decode_head_sha256"] for row in rows.values()}
    schedule_hashes = {row["schedule_sha256"] for row in rows.values()}
    head_counts = {
        row["parameters"]["decode_head_trainable"] for row in rows.values()
    }
    adaptation_counts = [
        rows[method]["parameters"]["backbone_trainable"]
        for method in ("rein", "cloud_adapter")
    ]
    method_param_ratio = max(adaptation_counts) / min(adaptation_counts)
    expected_val = {"img_path": "img_dir/val", "seg_map_path": "ann_dir/val"}
    expected_test = {"img_path": "img_dir/test", "seg_map_path": "ann_dir/test"}
    gates = {
        "identical_decoder": len(decoder_hashes) == 1,
        "identical_schedule": len(schedule_hashes) == 1,
        "identical_trainable_head_budget": len(head_counts) == 1,
        "fresh_initialization": all(row["load_from"] is None for row in rows.values()),
        "clean_val_split": all(row["val_prefix"] == expected_val for row in rows.values()),
        "sealed_test_split": all(row["test_prefix"] == expected_test for row in rows.values()),
        "forty_thousand_iterations": all(row["max_iters"] == 40000 for row in rows.values()),
        "frozen_baseline_has_no_backbone_training": rows["frozen"]["parameters"]["backbone_trainable"] == 0,
        "adaptation_methods_train_backbone_modules": all(value > 0 for value in adaptation_counts),
        "method_parameter_budget": method_param_ratio <= args.max_method_param_ratio,
    }
    result = {
        "phase": 29,
        "selection_split": "val",
        "test_evaluated": False,
        "max_method_param_ratio": args.max_method_param_ratio,
        "actual_method_param_ratio": method_param_ratio,
        "methods": rows,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
