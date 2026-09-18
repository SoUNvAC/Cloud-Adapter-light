import re


METRICS = ("aAcc", "mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall")
CLASS_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")


def parse_last_evaluation(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    aggregate_pattern = re.compile(
        r"\b(" + "|".join(METRICS) + r"):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))"
    )
    aggregate = None
    for line in text.splitlines():
        values = {
            name: float(value) for name, value in aggregate_pattern.findall(line)
        }
        if "mIoU" in values:
            aggregate = values
    if aggregate is None:
        raise RuntimeError(f"No mIoU metrics found in {log_path}")

    per_class = {}
    row_pattern = re.compile(
        r"\|\s*(clear|thick cloud|thin cloud|cloud shadow)\s*"
        r"\|\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\|"
    )
    for class_name, iou in row_pattern.findall(text):
        per_class[class_name] = {"IoU": float(iou)}
    return {"aggregate": aggregate, "per_class": per_class}
