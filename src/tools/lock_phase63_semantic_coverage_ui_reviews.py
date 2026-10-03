"""Freeze completed A/B UI-pilot review hashes before nominal strata are unsealed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from score_phase63_semantic_coverage_ui_pilot import freeze_reviews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--review-a", required=True)
    parser.add_argument("--review-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite frozen review lock: {output}")
    document = freeze_reviews(args.sealed, args.review_a, args.review_b)
    document["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(document, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
