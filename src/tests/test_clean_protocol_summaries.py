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
        "|    clear     | 86.0 | 90.0 |\n"
        "| thick cloud  | 82.0 | 88.0 |\n"
        "|  thin cloud  | 49.0 | 70.0 |\n"
        "| cloud shadow | 56.0 | 72.0 |\n"
        "Iter(test) [134/134] aAcc: 88.0 mIoU: "
        f"{miou:.3f} mAcc: 80.0 mDice: 81.0\n",
        encoding="utf-8",
    )
    return run_dir


class CleanProtocolSummaryTests(unittest.TestCase):
    def test_phase23_repair_sweep_selects_best_qualified_multiplier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for multiplier, miou in zip((1, 2, 3, 4), (67.70, 68.10, 68.10, 67.90)):
                run = root / f"mult{multiplier}_seed42"
                run.mkdir(parents=True)
                (run / "best_mIoU_iter_8000.pth").touch()
                (run / "VAL_EVAL_COMPLETE").touch()
                (run / "val_eval.log").write_text(
                    "| clear | 86.0 | 90.0 |\n"
                    f"Iter(test) [268/268] aAcc: 88.0 mIoU: {miou:.2f} "
                    "mAcc: 80.0 mDice: 81.0\n",
                    encoding="utf-8",
                )
            output = root / "summary.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "summarize_phase23_repair_sweep.py"),
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
            self.assertEqual(result["selected_multiplier"], 2)
            self.assertFalse(result["test_evaluated"])

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
            phase22_result = json.loads(phase22_output.read_text())
            self.assertTrue(phase22_result["passed"])
            self.assertEqual(
                phase22_result["runs"][0]["per_class"]["thin cloud"]["IoU"],
                49.0,
            )

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
            phase22_summary = temporary / "phase22.json"
            phase22_summary.write_text(
                json.dumps(
                    {
                        "phase": 22,
                        "passed": True,
                        "test_evaluated": False,
                        "runs": [
                            {"seed": 42, "validation": {"mIoU": 73.98}},
                            {"seed": 123, "validation": {"mIoU": 73.65}},
                            {"seed": 3407, "validation": {"mIoU": 73.78}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            root = temporary / "screen"
            values = {
                "baseline12": (73.98, 10.0),
                "blocks10": (68.5, 8.0),
                "blocks8": (64.0, 6.0),
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
                    "--phase22-summary",
                    str(phase22_summary),
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

    def test_phase25_paired_weak_class_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            seeds = (42, 123, 3407)
            phase23 = {
                "passed": True,
                "test_evaluated": False,
                "runs": [],
            }
            for seed, miou in zip(seeds, (67.5, 67.7, 67.6)):
                phase23["runs"].append(
                    {
                        "seed": seed,
                        "validation": {"mIoU": miou},
                        "per_class": {
                            "thin cloud": {"IoU": 49.0},
                            "cloud shadow": {"IoU": 56.0},
                        },
                    }
                )
            phase23_path = temporary / "phase23.json"
            phase23_path.write_text(json.dumps(phase23), encoding="utf-8")
            phase24_path = temporary / "phase24.json"
            phase24_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "test_evaluated": False,
                        "candidates": [
                            {
                                "name": "blocks10",
                                "qualifies": True,
                                "speedup": 1.2,
                                "benchmark": {
                                    "active_block_indices": [
                                        0,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        8,
                                        9,
                                        10,
                                        11,
                                    ]
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            root = temporary / "blocks10"
            for seed, miou in zip(seeds, (67.2, 67.4, 67.3)):
                make_run(root, seed, miou)
            output = root / "summary.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "summarize_phase25_block_kd.py"),
                    "--phase23-summary",
                    str(phase23_path),
                    "--phase24-summary",
                    str(phase24_path),
                    "--candidate",
                    "blocks10",
                    "--root",
                    str(root),
                    "--seeds",
                    *(str(seed) for seed in seeds),
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
            self.assertAlmostEqual(result["mean_paired_mIoU_drop_from_v12"], 0.3)
            self.assertAlmostEqual(
                result["mean_paired_weak_mIoU_drop_from_v12"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
