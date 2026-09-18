import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_run(root, method, seed, miou, thin_iou, shadow_iou):
    run_dir = root / method / f"seed{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "best_mIoU_iter_40000.pth").touch()
    (run_dir / "val_eval.log").write_text(
        "|    clear     | 86.0 | 90.0 |\n"
        "| thick cloud  | 82.0 | 88.0 |\n"
        f"|  thin cloud  | {thin_iou:.3f} | 70.0 |\n"
        f"| cloud shadow | {shadow_iou:.3f} | 72.0 |\n"
        "Iter(test) [134/134] aAcc: 88.0 mIoU: "
        f"{miou:.3f} mAcc: 80.0 mDice: 81.0\n",
        encoding="utf-8",
    )


class Phase29FairBaselineTests(unittest.TestCase):
    def test_vendored_rein_constructs_and_forwards_without_query_link(self):
        class Registry:
            def register_module(self):
                return lambda cls: cls

        mmseg = types.ModuleType("mmseg")
        models = types.ModuleType("mmseg.models")
        builder = types.ModuleType("mmseg.models.builder")
        builder.MODELS = Registry()
        previous = {
            name: sys.modules.get(name)
            for name in ("mmseg", "mmseg.models", "mmseg.models.builder")
        }
        sys.modules["mmseg"] = mmseg
        sys.modules["mmseg.models"] = models
        sys.modules["mmseg.models.builder"] = builder
        try:
            path = REPO_ROOT / "cloud_adapter" / "models" / "backbones" / "reins.py"
            spec = importlib.util.spec_from_file_location("phase29_reins_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            adapter = module.LoRAReins(
                num_layers=2,
                embed_dims=8,
                patch_size=2,
                token_length=4,
                link_token_to_query=False,
                lora_dim=2,
            )
            features = torch.randn(1, 5, 8)
            output = adapter(features, layer=0, batch_first=True, has_cls_token=True)
            self.assertEqual(tuple(output.shape), tuple(features.shape))
            self.assertFalse(hasattr(adapter, "transform"))
            self.assertTrue(hasattr(adapter, "mlp_delta_f"))
            self.assertTrue(hasattr(adapter, "mlp_token2feat"))
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def run_summary(self, root, output):
        audit = root / "audit.json"
        audit.write_text(
            json.dumps({"passed": True, "test_evaluated": False}),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "summarize_phase29_fair_baselines.py"),
                "--audit",
                str(audit),
                "--root",
                str(root),
                "--seeds",
                "42",
                "123",
                "3407",
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_paired_fair_baseline_gate_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds = (42, 123, 3407)
            for seed, delta in zip(seeds, (0.0, 0.1, -0.1)):
                make_run(root, "frozen", seed, 60.0 + delta, 40.0, 45.0)
                make_run(root, "rein", seed, 61.8 + delta, 42.0, 47.0)
                make_run(root, "cloud_adapter", seed, 62.0 + delta, 43.0, 48.0)
            output = root / "summary.json"
            process = self.run_summary(root, output)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(result["mean_cloud_gain_vs_frozen"], 2.0)
            self.assertAlmostEqual(result["mean_cloud_difference_vs_rein"], 0.2)
            self.assertAlmostEqual(
                result["mean_cloud_weak_gain_vs_frozen"], 3.0
            )
            self.assertFalse(result["test_evaluated"])

    def test_no_adapter_gain_triggers_stop_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (42, 123, 3407):
                make_run(root, "frozen", seed, 61.5, 42.0, 46.0)
                make_run(root, "rein", seed, 62.0, 43.0, 47.0)
                make_run(root, "cloud_adapter", seed, 61.8, 42.5, 46.5)
            output = root / "summary.json"
            process = self.run_summary(root, output)
            self.assertEqual(process.returncode, 1)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(result["gates"]["cloud_gain_vs_frozen"])
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
