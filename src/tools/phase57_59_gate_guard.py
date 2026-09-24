"""Hard prerequisite guard for Phases 57--59.

This file intentionally contains no ontology/M1/final-validation implementation:
the preregistered roadmap forbids developing or running those routes before all
earlier scientific gates pass.
"""

import argparse
import json
from pathlib import Path


PREREQUISITES = {
    57: Path("work_dirs/phase56_readout/summary.json"),
    58: Path("work_dirs/phase57_ontology_screen/summary.json"),
    59: Path("work_dirs/phase58_method_evolution/summary.json"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=(57, 58, 59), required=True)
    args = parser.parse_args()
    prerequisite = PREREQUISITES[args.phase]
    if not prerequisite.is_file():
        raise SystemExit(
            f"BLOCKED Phase {args.phase}: missing prerequisite {prerequisite}"
        )
    summary = json.loads(prerequisite.read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise SystemExit(
            f"BLOCKED Phase {args.phase}: prerequisite gate failed "
            f"({summary.get('decision', 'no decision')})"
        )
    print(json.dumps({
        "phase": args.phase,
        "prerequisite": str(prerequisite),
        "prerequisite_passed": True,
        "message": "Gate passed; a separately reviewed implementation is still required.",
    }, indent=2))


if __name__ == "__main__":
    main()
