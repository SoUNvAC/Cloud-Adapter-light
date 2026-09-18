from pathlib import Path
import sys
import unittest

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval_phase30_jetson import calculate_metrics, update_confusion  # noqa: E402
from audit_phase30_jetson import is_jetpack_6_1_or_later  # noqa: E402
from phase30_jetson_runtime import (  # noqa: E402
    clocks_are_locked,
    cuda_result,
    parse_tegrastats_line,
    percentile,
    summarize_power,
)


class Phase30JetsonProtocolTests(unittest.TestCase):
    def test_tegrastats_parser_extracts_target_signals(self):
        line = (
            "RAM 3124/7620MB (lfb 9x4MB) CPU [12%@1510] "
            "GR3D_FREQ 77% cpu@52.1C gpu@55.3C VDD_IN 12345mW/11000mW"
        )
        result = parse_tegrastats_line(line)
        self.assertEqual(result["ram_used_mb"], 3124)
        self.assertEqual(result["ram_total_mb"], 7620)
        self.assertEqual(result["gpu_utilization_percent"], 77)
        self.assertEqual(result["vdd_in_mw"], 12345)
        self.assertEqual(result["vdd_in_average_mw"], 11000)
        self.assertEqual(result["temperatures_c"]["gpu"], 55.3)

    def test_power_summary_uses_resident_idle_baseline(self):
        samples = [
            {"vdd_in_mw": 5000, "ram_used_mb": 2000, "temperatures_c": {"gpu": 40}},
            {"vdd_in_mw": 6000, "ram_used_mb": 2100, "temperatures_c": {"gpu": 41}},
            {"vdd_in_mw": 10000, "ram_used_mb": 3500, "temperatures_c": {"gpu": 60}},
            {"vdd_in_mw": 12000, "ram_used_mb": 3600, "temperatures_c": {"gpu": 62}},
        ]
        result = summarize_power(samples, idle_count=2, mean_latency_ms=50.0)
        self.assertEqual(result["idle_vdd_in_w"], 5.5)
        self.assertEqual(result["active_vdd_in_mean_w"], 11.0)
        self.assertEqual(result["dynamic_power_mean_w"], 5.5)
        self.assertAlmostEqual(result["total_energy_j_per_image"], 0.55)
        self.assertAlmostEqual(result["dynamic_energy_j_per_image"], 0.275)
        self.assertEqual(result["peak_ram_mb"], 3600)
        self.assertEqual(result["peak_temperature_c"], 62)

    def test_locked_clock_parser_requires_three_equal_ranges(self):
        locked = "\n".join(
            [
                "cpu0: Online=1 Governor=performance MinFreq=1510400 "
                "MaxFreq=1510400 CurrentFreq=1510400",
                "GPU: MinFreq=1020000000 MaxFreq=1020000000 "
                "CurrentFreq=1020000000",
                "EMC: MinFreq=3199000000 MaxFreq=3199000000 "
                "CurrentFreq=3199000000",
            ]
        )
        self.assertTrue(clocks_are_locked(locked))
        self.assertFalse(clocks_are_locked(locked.replace("MaxFreq=1510400", "MaxFreq=1200000")))
        self.assertFalse(clocks_are_locked("\n".join(locked.splitlines()[:2])))

    def test_jetpack_release_gate_rejects_6_0_and_future_major(self):
        self.assertTrue(
            is_jetpack_6_1_or_later("# R36 (release), REVISION: 4.3, GCID: 1")
        )
        self.assertFalse(
            is_jetpack_6_1_or_later("# R36 (release), REVISION: 3.0, GCID: 1")
        )
        self.assertFalse(
            is_jetpack_6_1_or_later("# R37 (release), REVISION: 1.0, GCID: 1")
        )

    def test_metrics_and_shape_guard(self):
        confusion = np.zeros((4, 4), dtype=np.int64)
        prediction = np.array([[0, 1], [2, 3]], dtype=np.int64)
        update_confusion(confusion, prediction, prediction.copy())
        metrics = calculate_metrics(confusion)
        self.assertEqual(metrics["mIoU"], 100.0)
        with self.assertRaises(ValueError):
            update_confusion(confusion, prediction, prediction[:, :1])

    def test_cuda_result_and_percentile_helpers(self):
        self.assertEqual(cuda_result((0, 1234), "fake"), 1234)
        self.assertIsNone(cuda_result(0, "fake"))
        with self.assertRaises(RuntimeError):
            cuda_result((2,), "fake")
        self.assertEqual(percentile([1.0, 2.0, 10.0], 0.9), 10.0)


if __name__ == "__main__":
    unittest.main()
