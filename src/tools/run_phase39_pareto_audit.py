import argparse
import json
import math
from pathlib import Path
import statistics


FILES = {
    phase: Path(f"work_dirs/phase{phase}_{suffix}/summary.json")
    for phase, suffix in {
        31: "mobilenetv2_litefpn",
        32: "l8_litefpn",
        33: "litefpn_3seed",
        34: "litefpn_kd",
        35: "bilitefpn",
        36: "mobilenetv3_litefpn",
        37: "weakclass_supervision",
        38: "boundary_supervision",
    }.items()
}
EXPECTED = {31: True, 32: True, 33: False, 34: False, 35: False,
            36: False, 37: False, 38: False}
PROTOCOL = {"precision": "fp16", "weight_dtype": "fp16",
            "batch_size": 1, "input_size": 512}


def weak(row):
    if "weak_class_mIoU" in row:
        return float(row["weak_class_mIoU"])
    classes = row["per_class"]
    return (float(classes["thin cloud"]["IoU"]) +
            float(classes["cloud shadow"]["IoU"])) / 2


def candidate_row(phase, row):
    bench = row["candidate_benchmark"]
    return {
        "phase": phase,
        "validation_mIoU": float(row["validation"]["mIoU"]),
        "weak_class_mIoU": weak(row),
        "speedup": float(row["speedup"]),
        "parameters_m": float(bench["parameters_m"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs/phase39_pareto_audit")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {phase: json.loads(path.read_text(encoding="utf-8"))
            for phase, path in FILES.items()}

    values = [float(run["validation"]["mIoU"])
              for run in data[33]["runs"]]
    stats_ok = (
        math.isclose(statistics.fmean(values), data[33]["mean_mIoU"], abs_tol=1e-12)
        and math.isclose(statistics.stdev(values), data[33]["std_mIoU"], abs_tol=1e-12)
    )
    benchmark_rows = [
        value[key]
        for phase, value in data.items()
        for key in ("baseline_benchmark", "candidate_benchmark")
        if key in value
    ]
    protocol_ok = all(all(row.get(k) == v for k, v in PROTOCOL.items())
                      for row in benchmark_rows)
    sealed = all(row.get("internal_test_evaluated") is False and
                 row.get("test_evaluated") is False for row in data.values())
    decisions_ok = all(data[phase].get("passed") is expected
                       for phase, expected in EXPECTED.items())

    rows = [candidate_row(phase, data[phase])
            for phase in (31, 34, 35, 36, 37, 38)]
    for row in rows:
        row["uniform_gates"] = {
            "validation_mIoU": row["validation_mIoU"] >= 68.0,
            "weak_class_mIoU": row["weak_class_mIoU"] >= 51.0,
            "native_fp16_speedup": row["speedup"] >= 2.0,
            "parameters_m": row["parameters_m"] <= 3.0,
        }
        row["uniform_pass"] = all(row["uniform_gates"].values())

    phase31_stable = data[33]["std_mIoU"] <= 0.75
    phase31_external = data[32]["passed"] is True
    qualified = any(row["uniform_pass"] for row in rows) and phase31_stable and phase31_external
    audit_gates = {
        "all_summaries_present": len(data) == 8,
        "recorded_decisions_reproduced": decisions_ok,
        "phase33_statistics_reproduced": stats_ok,
        "benchmark_protocol_consistent": protocol_ok,
        "internal_test_sealed": sealed,
        "finite_uniform_metrics": all(
            math.isfinite(row[key]) for row in rows
            for key in ("validation_mIoU", "weak_class_mIoU", "speedup", "parameters_m")
        ),
    }
    result = {
        "phase": 39,
        "audit_only": True,
        "internal_test_evaluated": False,
        "test_evaluated": False,
        "uniform_thresholds": {"min_mIoU": 68.0, "min_weak_mIoU": 51.0,
                               "min_speedup": 2.0, "max_parameters_m": 3.0,
                               "max_three_seed_std_mIoU": 0.75},
        "candidates": rows,
        "phase31_external_gate": phase31_external,
        "phase31_stability_gate": phase31_stable,
        "qualified_candidate": qualified,
        "audit_gates": audit_gates,
        "passed": all(audit_gates.values()),
        "decision": "frozen_stratified_error_audit" if not qualified else "external_confirmation",
        "closed_directions": ["class_weight_tuning", "auxiliary_supervision_losses",
                              "MobileNet_family_swap", "LiteFPN_fusion", "teacher_KD"],
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "Phase 39 frozen Pareto and evidence audit",
        "=========================================",
        "",
        "phase  val_mIoU  weak_mIoU  speedup  params_M  uniform_pass",
    ]
    lines += [f"{r['phase']:>5}  {r['validation_mIoU']:>8.2f}  "
              f"{r['weak_class_mIoU']:>9.3f}  {r['speedup']:>7.3f}  "
              f"{r['parameters_m']:>8.3f}  {str(r['uniform_pass']):>12}"
              for r in rows]
    lines += ["", f"Phase 31 external gate: {phase31_external}",
              f"Phase 31 stability gate: {phase31_stable}",
              f"Qualified candidate: {qualified}",
              f"Audit integrity passed: {result['passed']}",
              "Next: frozen stratified error audit; internal test remains sealed."]
    (root / "PHASE39_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
