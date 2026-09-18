import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_run(root, seed, miou):
    run_dir = root / f"seed{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "best_mIoU_iter_40000.pth").touch()
    (run_dir / "val_eval.log").write_text(
        "Iter(test) [134/134] aAcc: 88.0 mIoU: "
        f"{miou:.3f} mAcc: 80.0 mDice: 81.0\n",
        encoding="utf-8",
    )
    return run_dir


class CleanProtocolSummaryTests(unittest.TestCase):
    def test_phase22_pass_and_phase23_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            seeds = (42, 123, 3407)
            phase22_root = temporary / "phase22"
            for seed, miou in zip(seeds, (70.0, 70.2, 70.1)):
                make_run(phase22_root, seed, miou)
            phase22_output = phase22_root / "summary.json"
            phase22 = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "summarize_phase22_clean_v8.py"),
                    "--root",
                    str(phase22_root),
                    "--seeds",
                    *(str(seed) for seed in seeds),
                    "--output",
                    str(phase22_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(phase22.returncode, 0, phase22.stderr)
            self.assertTrue(json.loads(phase22_output.read_text())["passed"])

            phase23_root = temporary / "phase23"
            for seed, miou in zip(seeds, (67.5, 67.8, 67.6)):
                make_run(phase23_root, seed, miou)
            phase23_output = phase23_root / "summary.json"
            phase23 = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "summarize_phase23_clean_v12.py"),
                    "--phase22-summary",
                    str(phase22_output),
                    "--root",
                    str(phase23_root),
                    "--seeds",
                    *(str(seed) for seed in seeds),
                    "--output",
                    str(phase23_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(phase23.returncode, 0, phase23.stderr)
            result = json.loads(phase23_output.read_text())
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(result["mean_paired_mIoU_drop_from_v8"], 2.4666667)

    def test_phase22_gate_failure_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds = (42, 123, 3407)
            for seed, miou in zip(seeds, (68.0, 70.0, 70.0)):
                make_run(root, seed, miou)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "summarize_phase22_clean_v8.py"),
                    "--root",
                    str(root),
                    "--seeds",
                    *(str(seed) for seed in seeds),
                    "--output",
                    str(root / "summary.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 1)
            self.assertFalse(json.loads((root / "summary.json").read_text())["passed"])

    def test_phase24_selects_only_qualified_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            phase23_summary = temporary / "phase23.json"
            phase23_summary.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "test_evaluated": False,
                        "runs": [
                            {"seed": 42, "validation": {"mIoU": 67.5}},
                            {"seed": 123, "validation": {"mIoU": 67.6}},
                            {"seed": 3407, "validation": {"mIoU": 67.7}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            root = temporary / "screen"
            values = {
                "baseline12": (67.5, 10.0),
                "blocks10": (62.5, 8.0),
                "blocks8": (58.0, 6.0),
                "blocks6": (50.0, 5.0),
                "blocks4": (40.0, 4.0),
            }
            for name, (miou, latency) in values.items():
                candidate = root / name
                candidate.mkdir(parents=True)
                (candidate / "val_eval.log").write_text(
                    f"Iter(test) [134/134] mIoU: {miou}\n", encoding="utf-8"
                )
                (candidate / "benchmark.json").write_text(
                    json.dumps({"latency_mean_ms": latency}), encoding="utf-8"
                )
            output = root / "summary.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "summarize_phase24_block_screen.py"),
                    "--phase23-summary",
                    str(phase23_summary),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["selected_candidates"], ["blocks10"])


if __name__ == "__main__":
    unittest.main()
