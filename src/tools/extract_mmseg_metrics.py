import argparse
from pathlib import Path
import re


METRICS = ("aAcc", "mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract final MMSeg metrics from a log")
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-format", choices=("text", "tsv"), default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log)
    checkpoint_path = Path(args.checkpoint)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metric_pattern = re.compile(
        r"\b(" + "|".join(METRICS) + r"):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))"
    )

    result = None
    for line in reversed(text.splitlines()):
        values = dict(metric_pattern.findall(line))
        if "mIoU" in values:
            result = values
            break
    if result is None:
        raise RuntimeError(f"No final mIoU metric found in {log_path}")

    size_mib = checkpoint_path.stat().st_size / 2**20
    values = [result.get(metric, "-") for metric in METRICS]
    if args.output_format == "tsv":
        print("\t".join([f"{size_mib:.2f}", *values]))
    else:
        print(f"Checkpoint size: {size_mib:.2f} MiB")
        for metric, value in zip(METRICS, values):
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
