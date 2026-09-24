"""Create the machine-checkable Phase 56 gate summary."""

import argparse
import json
from pathlib import Path

from phase56_protocol import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="work_dirs/phase56_readout/feature_cache_manifest.json")
    parser.add_argument("--readout", default="work_dirs/phase56_readout/readout_summary.json")
    parser.add_argument("--support", default="work_dirs/phase56_readout/label_support_summary.json")
    parser.add_argument("--dense", default="work_dirs/phase56_readout/dense_readout_summary.json")
    parser.add_argument("--output", default="work_dirs/phase56_readout/summary.json")
    args = parser.parse_args()
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    readout = json.loads(Path(args.readout).read_text(encoding="utf-8"))
    support = json.loads(Path(args.support).read_text(encoding="utf-8"))
    dense = json.loads(Path(args.dense).read_text(encoding="utf-8"))
    summary = {
        "phase": 56,
        "objective": "Test whether a parameter-limited readout can recover four-class prediction from frozen Phase52A features.",
        "artifacts": {
            "feature_cache_manifest": {"path": args.cache, "sha256": sha256(args.cache)},
            "readout_summary": {"path": args.readout, "sha256": sha256(args.readout)},
            "label_support_summary": {"path": args.support, "sha256": sha256(args.support)},
            "dense_readout_summary": {"path": args.dense, "sha256": sha256(args.dense)},
        },
        "protocol": cache["protocol"],
        "best_parameter_limited_readout": readout["best_parameter_limited_readout"],
        "dense_target_val_metrics": dense["metrics"],
        "gates": dense["gates"],
        "passed": dense["passed"],
        "decision": dense["decision"],
        "label_support_claim": support["claims"],
        "network": {
            "required_during_experiment": False,
            "note": "All datasets/checkpoints were already local; only Git SSH was used for synchronization.",
        },
        "stop_line": "Do not implement or run Phase 57 if passed is false.",
        "target_test_evaluated": False,
        "cloudsen_internal_test_evaluated": False,
    }
    output = Path(args.output)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
