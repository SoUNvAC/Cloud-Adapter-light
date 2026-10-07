import unittest
from cleanup_phase65_archived_results import selection, SOURCE


class PolicyTests(unittest.TestCase):
    def test_protected_checkpoint_and_lineage(self):
        self.assertFalse(selection(SOURCE))
        self.assertFalse(selection('src/work_dirs/phase65a/split_lock.json'))
        self.assertFalse(selection('src/work_dirs/phase65a/geospatial_python/cache.npy'))

    def test_historical_artifacts_and_backup(self):
        self.assertTrue(selection('src/work_dirs/phase64_target_parent/run/best.pth'))
        self.assertTrue(selection('src/work_dirs/phase54_information_audit/cache/a.npz'))
        self.assertTrue(selection('src/result_backups/phase65_20261006/all_work_dirs.tar'))
        self.assertFalse(selection('src/result_backups/phase65_20261006/all_work_dirs.tar.json'))

    def test_preserve_records_and_outside_data(self):
        for path in ['src/work_dirs/phase54/cache_summary.json', 'src/work_dirs/phase64/a.csv',
                     'src/work_dirs/phase64/console.log', 'data/source.npy']:
            self.assertFalse(selection(path))


if __name__ == '__main__':
    unittest.main()
