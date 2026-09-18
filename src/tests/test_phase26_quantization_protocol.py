import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from quantize_phase26_int8_qdq import select_evenly_spaced  # noqa: E402


class Phase26ProtocolTests(unittest.TestCase):
    def test_calibration_sampler_is_deterministic_and_spans_validation(self):
        paths = [Path(f"tile_{index:04d}.png") for index in range(535)]
        selected = select_evenly_spaced(paths, 256)
        self.assertEqual(len(selected), 256)
        self.assertEqual(len(set(selected)), 256)
        self.assertEqual(selected[0], paths[0])
        self.assertEqual(selected[-1], paths[-1])
        self.assertEqual(selected, select_evenly_spaced(paths, 256))

    def test_checkpoint_rule_uses_median_validation_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "blocks10"
            candidate.mkdir()
            summary = {
                "passed": True,
                "test_evaluated": False,
                "candidate": "blocks10",
                "active_block_indices": [0, 2, 3, 4, 5, 6, 8, 9, 10, 11],
                "runs": [
                    {
                        "seed": 42,
                        "checkpoint": "seed42/best.pth",
                        "validation": {"mIoU": 67.1},
                    },
                    {
                        "seed": 123,
                        "checkpoint": "seed123/best.pth",
                        "validation": {"mIoU": 67.8},
                    },
                    {
                        "seed": 3407,
                        "checkpoint": "seed3407/best.pth",
                        "validation": {"mIoU": 67.4},
                    },
                ],
            }
            (candidate / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            output = root / "selection.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "select_phase26_checkpoint.py"),
                    "--phase25-root",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["seed"], 3407)
            self.assertEqual(result["reference_validation_mIoU"], 67.4)
            self.assertFalse(result["test_evaluated"])

    def test_selection_rejects_multiple_passed_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("blocks10", "blocks8"):
                candidate = root / name
                candidate.mkdir()
                (candidate / "summary.json").write_text(
                    json.dumps({"passed": True, "test_evaluated": False}),
                    encoding="utf-8",
                )
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "select_phase26_checkpoint.py"),
                    "--phase25-root",
                    str(root),
                    "--output",
                    str(root / "selection.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
