import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


EXPECTED = {
    "phase22": ("work_dirs/phase22_clean_v8/summary.json", True),
    "phase23": ("work_dirs/phase23_clean_v12/summary.json", False),
    "phase23_repair": ("work_dirs/phase23_repair_lrsweep/summary.json", False),
    "phase24": ("work_dirs/phase24_block_screen/summary.json", False),
    "phase25": ("work_dirs/phase25_mlp_screen/summary.json", False),
    "phase26": ("work_dirs/phase26_resnet18_pilot/summary.json", True),
    "phase27": ("work_dirs/phase27_resnet18_3seed/summary.json", True),
    "phase28": ("work_dirs/phase28_l8_external/summary.json", False),
    "phase29": ("work_dirs/phase29_mobilenetv2_pilot/summary.json", False),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats_match(summary):
    values = [float(run["validation"]["mIoU"]) for run in summary["runs"]]
    return (
        abs(statistics.fmean(values) - float(summary["mean_mIoU"])) < 1e-9
        and abs(statistics.stdev(values) - float(summary["std_mIoU"])) < 1e-9
    )


def benchmark_protocol(row):
    return all(
        row[key] == value
        for key, value in {
            "precision": "fp16",
            "weight_dtype": "fp16",
            "batch_size": 1,
            "input_size": 512,
        }.items()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs/phase30_final_audit")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    phase21_path = Path("eval_result/phase21_protocol/result.json")
    phase21 = json.loads(phase21_path.read_text(encoding="utf-8"))
    summaries = {}
    status_matches = {}
    hashes = {phase21_path.as_posix(): sha256(phase21_path)}
    for name, (path_string, expected_status) in EXPECTED.items():
        path = Path(path_string)
        row = json.loads(path.read_text(encoding="utf-8"))
        summaries[name] = row
        status_matches[name] = row.get("passed") is expected_status
        hashes[path.as_posix()] = sha256(path)

    test_sealed = all(
        row.get("test_evaluated") is False for row in summaries.values()
    ) and all(
        summaries[name].get("internal_test_evaluated") is False
        for name in ("phase28", "phase29")
    )
    referenced_checkpoints = [
        Path(run["checkpoint"])
        for phase in (summaries["phase22"], summaries["phase27"])
        for run in phase["runs"]
    ]
    checkpoints_exist = all(path.is_file() for path in referenced_checkpoints)
    phase26 = summaries["phase26"]
    phase29 = summaries["phase29"]
    benchmark_consistent = all(
        benchmark_protocol(row)
        for row in (
            phase26["baseline_benchmark"],
            phase26["candidate_benchmark"],
            phase29["baseline_benchmark"],
            phase29["candidate_benchmark"],
        )
    )
    tracked_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gates = {
        "phase21_passed": phase21.get("phase") == 21 and phase21.get("passed") is True,
        "phase_statuses_match": all(status_matches.values()),
        "phase22_statistics_recomputed": stats_match(summaries["phase22"]),
        "phase27_statistics_recomputed": stats_match(summaries["phase27"]),
        "referenced_checkpoints_exist": checkpoints_exist,
        "internal_test_sealed": test_sealed,
        "benchmark_protocol_consistent": benchmark_consistent,
        "tracked_worktree_clean": tracked_status == "",
    }
    result = {
        "phase": 30,
        "audit_only": True,
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "git_commit": commit,
        "summary_sha256": hashes,
        "expected_statuses": {
            name: expected for name, (_, expected) in EXPECTED.items()
        },
        "observed_statuses": {
            name: row.get("passed") for name, row in summaries.items()
        },
        "gates": gates,
        "passed": all(gates.values()),
        "final_decision": {
            "accuracy_anchor": "Phase 22 V8",
            "qualified_compressed_successor": None,
            "resnet18": "rejected by Phase 28 external gates",
            "mobilenetv2": "rejected by Phase 29 external absolute and speed gates",
            "internal_test_action": "remains sealed",
        },
    }
    (root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    p22 = summaries["phase22"]
    p27 = summaries["phase27"]
    p28 = summaries["phase28"]
    report = f"""# Phase 30 final frozen evidence report

Audit commit: `{commit}`

## Supported conclusions

- Clean accuracy anchor: Phase 22 V8, {p22['mean_mIoU']:.4f} +/- {p22['std_mIoU']:.4f} validation mIoU across three seeds.
- Internal efficiency point: Phase 27 ResNet-18, {p27['mean_mIoU']:.4f} +/- {p27['std_mIoU']:.4f} validation mIoU, {p27['frozen_speedup']:.3f}x native-FP16 speedup, {p27['frozen_parameters_m']:.3f}M parameters.
- External L8 evidence: V8 mean {p28['models']['v8']['mean_mIoU']:.4f} mIoU; ResNet-18 mean {p28['models']['resnet18']['mean_mIoU']:.4f} mIoU and failed Phase 28.
- MobileNetV2 seed-42 reached {phase29['validation']['mIoU']:.2f} internal-val mIoU but only {phase29['external']['mIoU']:.2f} L8 mIoU and {phase29['speedup']:.3f}x speedup, so it failed Phase 29.

## Claims not supported

- No compressed successor passed the joint clean-validation, external-generalization, and real-latency gates.
- ResNet-18 must not be presented as externally qualified despite its strong internal stability and speed.
- MobileNetV2 must not be presented as a 2x deployment result.
- Landsat-8 Biome was present in the historical workspace and is not a pristine never-observed holdout.
- CloudSEN internal test remains sealed; there is no post-Phase-21 internal-test claim.

## Audit outcome

All evidence-integrity gates passed: **{all(gates.values())}**. This validates consistency, not model performance. A TGRS submission still needs a genuinely new external/geographic holdout and a method that passes it before final internal-test unsealing.
"""
    (root / "FINAL_EVIDENCE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
