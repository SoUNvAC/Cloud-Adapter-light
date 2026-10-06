"""Meaningful fail-closed protocol checks; synthetic fixtures are not results."""
import tempfile
import unittest
from pathlib import Path
from phase65a_freeze import freeze
from phase65b_audit_matched import audit, KEYS
from phase65c_gate import validate
from backup_phase65 import pack, verify


class ProtocolTests(unittest.TestCase):
    def fixture(self):
        scenes = [f's{i:02}' for i in range(16)]
        rows = [dict(scene=s, new_split='target_train', usgs_shadows='yes') for s in scenes]
        lineage = dict(used_scenes=['historical'], sealed_scenes=['sealed'],
                       unused_verified_scenes=scenes, provenance_sources=['audit'])
        return rows, lineage

    def test_scene_split_and_shadow_filter(self):
        rows, lineage = self.fixture()
        result = freeze(rows, lineage)
        self.assertEqual(sum(r['phase65_split'] == 'confirmation' for r in result), 8)
        rows[0]['usgs_shadows'] = 'no'
        with self.assertRaises(ValueError):
            freeze(rows, lineage)

    def test_historical_and_sealed_overlap_rejected(self):
        rows, lineage = self.fixture()
        for key in ('used_scenes', 'sealed_scenes'):
            original = lineage[key]
            lineage[key] = ['s00']
            with self.assertRaises(ValueError):
                freeze(rows, lineage)
            lineage[key] = original

    def test_matching_cannot_accept_confirmation_or_scale_change(self):
        rgb = {key: 'same' for key in KEYS}
        rgb.update(record_id='one', split='fit')
        six = rgb.copy()
        self.assertFalse(audit(dict(rgb=[rgb], six=[six]))['training_authorized'])
        six['resolution'] = 'different'
        with self.assertRaises(ValueError):
            audit(dict(rgb=[rgb], six=[six]))
        six = rgb.copy()
        rgb['split'] = six['split'] = 'confirmation'
        with self.assertRaises(ValueError):
            audit(dict(rgb=[rgb], six=[six]))

    def test_failed_hypothesis_cannot_progress(self):
        a = dict(status='complete', h1_r_ci95=[-0.1, 0.6], h1_prediction_rmse=1,
                 constant_rmse=2, h2_auc_ci95=[0.6, 0.9])
        b = dict(status='complete')
        with self.assertRaises(ValueError):
            validate(a, b)

    def test_backup_roundtrip_and_corruption(self):
        workspace = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=workspace) as tmp:
            base = Path(tmp)
            source = base / 'work_dirs'
            source.mkdir()
            (source / 'actual.log').write_text('fixture only')
            tar = base / 'backup.tar'
            pack(source, tar)
            self.assertEqual(verify(tar)['status'], 'verified')
            with tar.open('r+b') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError):
                verify(tar)


if __name__ == '__main__':
    unittest.main()
