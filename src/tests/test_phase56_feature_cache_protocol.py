from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.phase56_protocol import deterministic_samples, sample_spatial


def test_deterministic_stratified_sampling_is_capped_and_repeatable():
    mask = np.asarray([[0, 0, 1, 1], [2, 2, 3, 3], [0, 1, 2, 3]])
    first_coordinates, first_labels = deterministic_samples(mask, "image", 2, 56)
    second_coordinates, second_labels = deterministic_samples(mask, "image", 2, 56)
    np.testing.assert_array_equal(first_coordinates, second_coordinates)
    np.testing.assert_array_equal(first_labels, second_labels)
    assert np.bincount(first_labels, minlength=4).tolist() == [2, 2, 2, 2]


def test_sample_spatial_maps_original_coordinates_to_feature_cells():
    import torch

    feature = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    coordinates = np.asarray([[0, 0], [0, 3], [3, 0], [3, 3]])
    sampled = sample_spatial(feature, coordinates, (4, 4)).cpu().numpy().reshape(-1)
    np.testing.assert_array_equal(sampled, np.asarray([0, 1, 2, 3]))


def test_phase56_launcher_enforces_requested_environment_and_gpu_lock():
    text = (REPO_ROOT / "tools/run_phase56_readout_4090d.sh").read_text(encoding="utf-8")
    assert "conda activate cloud-lite-pt210" in text
    assert "flock -n" in text
    assert "cache_phase56_readout_features.py" in text
    assert "evaluate_phase56_readout_dense.py" in text


if __name__ == "__main__":
    test_deterministic_stratified_sampling_is_capped_and_repeatable()
    test_sample_spatial_maps_original_coordinates_to_feature_cells()
    test_phase56_launcher_enforces_requested_environment_and_gpu_lock()
    print("Phase 56 feature-cache protocol tests passed")
