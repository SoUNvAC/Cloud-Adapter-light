import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


FILES = {
    31: "phase31_mobilenetv2_litefpn",
    32: "phase32_l8_litefpn",
    33: "phase33_litefpn_3seed",
    34: "phase34_litefpn_kd",
    35: "phase35_bilitefpn",
    36: "phase36_mobilenetv3_litefpn",
    37: "phase37_weakclass_supervision",
    38: "phase38_boundary_supervision",
    39: "phase39_pareto_audit",
    40: "phase40_error_audit",
    41: "phase41_detail_litefpn",
    42: "phase42_weak_sampling",
    43: "phase43_context_litefpn",
}
EXPECTED = {31: True, 32: True, 33: False, 34: False, 35: False,
            36: False, 37: False, 38: False, 39: True, 40: True,
            41: False, 42: False, 43: False}
MODEL_PHASES = (31, 34, 35, 36, 37, 38, 41, 42, 43)
PROTOCOL = {"precision": "fp16", "weight_dtype": "fp16",
            "batch_size": 1, "input_size": 512}


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def weak(row):
    if "weak_class_mIoU" in row:
        return float(row["weak_class_mIoU"])
    values = row["per_class"]
    return statistics.fmean((values["thin cloud"]["IoU"],
                             values["cloud shadow"]["IoU"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs/phase44_final_audit")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {phase: Path("work_dirs") / name / "summary.json"
             for phase, name in FILES.items()}
    data = {phase: json.loads(path.read_text(encoding="utf-8"))
            for phase, path in paths.items()}
    hashes = {path.as_posix(): sha256(path) for path in paths.values()}

    values = [float(run["validation"]["mIoU"])
              for run in data[33]["runs"]]
    statistics_ok = (
        math.isclose(statistics.fmean(values), data[33]["mean_mIoU"], abs_tol=1e-12)
        and math.isclose(statistics.stdev(values), data[33]["std_mIoU"], abs_tol=1e-12)
    )
    decisions_ok = all(data[phase].get("passed") is expected
                       for phase, expected in EXPECTED.items())
    sealed = all(row.get("internal_test_evaluated") is False and
                 row.get("test_evaluated") is False for row in data.values())
    benchmarks = [data[phase][key] for phase in MODEL_PHASES
                  for key in ("baseline_benchmark", "candidate_benchmark")]
    protocol_ok = all(all(row.get(key) == value for key, value in PROTOCOL.items())
                      for row in benchmarks)
    candidates = []
    for phase in MODEL_PHASES:
        row = data[phase]
        candidate = row["candidate_benchmark"]
        record = {"phase": phase,
                  "validation_mIoU": float(row["validation"]["mIoU"]),
                  "weak_class_mIoU": weak(row),
                  "speedup": float(row["speedup"]),
                  "parameters_m": float(candidate["parameters_m"])}
        record["uniform_pass"] = (
            record["validation_mIoU"] >= 68.0
            and record["weak_class_mIoU"] >= 51.0
            and record["speedup"] >= 2.0
            and record["parameters_m"] <= 3.0
        )
        candidates.append(record)
    phase31_stable = float(data[33]["std_mIoU"]) <= 0.75
    phase31_external = data[32].get("passed") is True
    qualified = any(row["uniform_pass"] for row in candidates)
    qualified = qualified and phase31_stable and phase31_external
    best = max(candidates, key=lambda row: row["validation_mIoU"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"], check=True,
        capture_output=True, text=True).stdout.strip()
    gates = {
        "all_summaries_present": len(data) == 13,
        "recorded_decisions_reproduced": decisions_ok,
        "internal_test_sealed": sealed,
        "benchmark_protocol_consistent": protocol_ok,
        "phase33_statistics_reproduced": statistics_ok,
        "finite_candidate_metrics": all(math.isfinite(value)
            for row in candidates for key, value in row.items()
            if key not in {"phase", "uniform_pass"}),
        "tracked_worktree_clean": tracked == "",
    }
    result = {
        "phase": 44,
        "audit_only": True,
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "git_commit": commit,
        "summary_sha256": hashes,
        "expected_statuses": EXPECTED,
        "observed_statuses": {phase: data[phase].get("passed") for phase in data},
        "uniform_thresholds": {"min_mIoU": 68.0, "min_weak_mIoU": 51.0,
                               "min_speedup": 2.0, "max_parameters_m": 3.0,
                               "max_three_seed_std_mIoU": 0.75},
        "candidates": candidates,
        "best_compact_validation_point": best,
        "phase31_external_gate": phase31_external,
        "phase31_stability_gate": phase31_stable,
        "qualified_candidate": qualified,
        "audit_gates": gates,
        "passed": all(gates.values()),
        "final_decision": "close_compact_phase31_43_program" if not qualified
                          else "continue_external_confirmation",
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = (
        "Phase 44 final frozen evidence audit\n"
        "====================================\n\n"
        f"Audit integrity passed: {result['passed']}\n"
        f"Qualified compact candidate: {qualified}\n"
        f"Best compact validation point: Phase {best['phase']}, "
        f"{best['validation_mIoU']:.2f} mIoU, "
        f"{best['weak_class_mIoU']:.3f} weak mIoU, {best['speedup']:.3f}x.\n"
        f"Phase 31 stability gate: {phase31_stable}\n"
        f"Phase 31 external gate: {phase31_external}\n"
        "CloudSEN internal test remains sealed.\n"
        "Decision: close the Phase 31-43 compact program; require a new core "
        "architecture or pristine external/geographic data.\n"
    )
    (root / "PHASE44_REPORT.txt").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
