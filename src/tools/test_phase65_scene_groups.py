"""Check scene independence and conservative exclusion, using synthetic IDs."""
import unittest
from prepare_phase65_scene_groups import group_products


def row(date, tile, shadows='1'):
    return dict(scene=f'S2A_MSIL1C_{date}T010721_N0206_R045_{tile}_{date}T041600',
                shadows_marked=shadows)


class SceneGroupsTests(unittest.TestCase):
    def test_transitive_source_exclusion(self):
        # A/B share tile; B/C share date+orbit. All must be one excluded group.
        tags = [row('20180101', 'T53HLD'), row('20180102', 'T53HLD'),
                row('20180102', 'T54HLD')]
        groups = group_products(tags, {'T54HLD'}, set(), set())
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['members']), 3)
        self.assertFalse(groups[0]['eligible'])

    def test_invalid_shadow_connector_is_never_sample(self):
        tags = [row('20180101', 'T53HLD'), row('20180102', 'T53HLD', '0'),
                row('20180102', 'T54HLD')]
        groups = group_products(tags, set(), set(), set())
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['shadow_valid_products']), 2)
        self.assertNotIn(tags[1]['scene'], groups[0]['shadow_valid_products'])

    def test_stable_group_and_historical_exclusion(self):
        tags = [row('20180101', 'T53HLD'), row('20180102', 'T54HLD')]
        historical = {tags[0]['scene']}
        forward = group_products(tags, set(), set(), historical)
        reverse = group_products(tags[::-1], set(), set(), historical)
        self.assertEqual(forward, reverse)
        self.assertEqual(sum(g['eligible'] for g in forward), 1)


if __name__ == '__main__':
    unittest.main()
