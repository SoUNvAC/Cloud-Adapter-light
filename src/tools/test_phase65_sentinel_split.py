"""Independence and class-support gates for the pre-analysis split."""
import unittest
from freeze_phase65_sentinel_split import make_split


def fixture():
    tags = [dict(scene=f'product{i}', shadows_marked='1', shadow_percent='1') for i in range(260)]
    groups = [dict(group_id=f'{i:064x}', members=[r['scene']], shadow_valid_products=[r['scene']],
                   eligible=True, exclusion_reasons=[]) for i, r in enumerate(tags)]
    return groups, tags


class SplitTests(unittest.TestCase):
    def test_independence_and_reserved_cohorts(self):
        groups, tags = fixture()
        result = make_split(groups, tags)
        self.assertEqual(len({r['representative'] for r in result}), 260)
        for split in ('65a_confirmation', '65b_confirmation', '65c_final'):
            self.assertEqual(sum(r['split'] == split for r in result), 64)
        self.assertEqual(make_split(groups[::-1], tags[::-1]), result)

    def test_duplicate_group_members_stop(self):
        groups, tags = fixture()
        groups[1]['members'] = groups[0]['members']
        groups[1]['shadow_valid_products'] = groups[0]['shadow_valid_products']
        with self.assertRaises(ValueError):
            make_split(groups, tags)

    def test_no_h1_support_cannot_reselect(self):
        groups, tags = fixture()
        for row in tags:
            row['shadow_percent'] = '0'
        with self.assertRaises(ValueError):
            make_split(groups, tags)


if __name__ == '__main__':
    unittest.main()
