"""Plot the Phase 64 target-gain/source-retention Pareto evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METHODS = ("shared_parent", "full", "lora", "msre")
LABELS = {
    "shared_parent": "Shared parent",
    "full": "Full fine-tuning",
    "lora": "LoRA",
    "msre": "MsRE",
}
COLORS = {
    "shared_parent": "#0072B2",
    "full": "#D55E00",
    "lora": "#009E73",
    "msre": "#CC79A7",
}
MARKERS = {"shared_parent": "o", "full": "X", "lora": "s", "msre": "D"}


def rows_from_summary(summary: dict) -> list[dict]:
    if summary.get("input") != "RGB" or summary.get("seed") != 64:
        raise ValueError("Expected the frozen Phase64 RGB seed64 summary")
    rows = []
    for target in ("l8", "sparcs"):
        for method in METHODS:
            record = summary["methods"][target][method]
            if not record.get("complete") or not record.get("losses_finite"):
                raise RuntimeError(f"Incomplete or non-finite record: {target}/{method}")
            row = {
                "target": target,
                "method": method,
                "target_gain_miou": float(record["target_delta_mIoU"]),
                "source_retention_miou": float(record["source_retention"]["mIoU"]),
                "source_delta_miou": float(record["source_delta_mIoU"]),
                "trainable_parameters": int(record["trainable_parameters"]),
            }
            if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise RuntimeError(f"Non-finite plot value: {row}")
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def marker_size(parameters: int, minimum: int, maximum: int) -> float:
    low, high = math.log10(minimum), math.log10(maximum)
    fraction = (math.log10(parameters) - low) / (high - low)
    return 55.0 + 150.0 * fraction


def nondominated(rows: list[dict]) -> set[str]:
    result = set()
    for candidate in rows:
        dominated = any(
            other["method"] != candidate["method"]
            and other["target_gain_miou"] >= candidate["target_gain_miou"]
            and other["source_retention_miou"] >= candidate["source_retention_miou"]
            and (
                other["target_gain_miou"] > candidate["target_gain_miou"]
                or other["source_retention_miou"] > candidate["source_retention_miou"]
            )
            for other in rows
        )
        if not dominated:
            result.add(candidate["method"])
    return result


def plot(rows: list[dict], source_miou: float, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    minimum = min(row["trainable_parameters"] for row in rows)
    maximum = max(row["trainable_parameters"] for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True, sharey=True)
    offsets = {
        "l8": {
            "shared_parent": (6, -15),
            "full": (5, -11),
            "lora": (-39, 8),
            "msre": (5, 5),
        },
        "sparcs": {
            "shared_parent": (5, 5),
            "full": (5, -11),
            "lora": (5, -16),
            "msre": (5, 5),
        },
    }
    for ax, target in zip(axes, ("l8", "sparcs")):
        target_rows = [row for row in rows if row["target"] == target]
        frontier = nondominated(target_rows)
        ax.axhline(source_miou, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
        ax.axvline(0.0, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=0)
        for row in target_rows:
            size = marker_size(row["trainable_parameters"], minimum, maximum)
            edge = "black" if row["method"] in frontier else "white"
            width = 1.1 if row["method"] in frontier else 0.6
            ax.scatter(
                row["target_gain_miou"],
                row["source_retention_miou"],
                s=size,
                marker=MARKERS[row["method"]],
                color=COLORS[row["method"]],
                edgecolor=edge,
                linewidth=width,
                alpha=0.92,
                zorder=3,
            )
            ax.annotate(
                LABELS[row["method"]],
                (row["target_gain_miou"], row["source_retention_miou"]),
                xytext=offsets[target][row["method"]],
                textcoords="offset points",
                fontsize=6.5,
            )
        ax.set_title("L8 Biome" if target == "l8" else "SPARCS")
        ax.set_xlabel("Target mIoU gain (percentage points)")
        ax.set_xlim(-8.5, 14.0)
        ax.set_ylim(0.0, 82.5)
        ax.grid(axis="both", color="#E5E5E5", linewidth=0.5, zorder=-1)
    axes[0].set_ylabel("Retained CloudSEN source mIoU (%)")
    parameter_examples = (0.3, 1.3, 23.6)
    size_handles = [
        plt.scatter(
            [], [],
            s=marker_size(int(value * 1_000_000), minimum, maximum),
            facecolor="#BDBDBD", edgecolor="white", label=f"{value:g} M"
        )
        for value in parameter_examples
    ]
    axes[1].legend(
        handles=size_handles,
        title="Trainable params\n(log-scaled area)",
        frameon=False,
        loc="lower right",
    )
    fig.text(
        0.5,
        0.005,
        "Dashed line: unadapted source mIoU; black outline: non-dominated target-retention point.",
        ha="center",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"phase64_rgb_pareto.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = rows_from_summary(summary)
    out_dir = Path(args.out_dir)
    write_csv(rows, out_dir / "phase64_rgb_pareto.csv")
    if not args.csv_only:
        plot(rows, float(summary["source_only"]["source_mIoU"]), out_dir)


if __name__ == "__main__":
    main()
