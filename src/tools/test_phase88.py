"""Negative tests: incomplete or non-human submissions must never open B."""
import unittest
from unittest.mock import patch
from pathlib import Path
import prepare_phase88 as p


class HumanGate(unittest.TestCase):
    def reject(self, packet):
        # Synthetic data is memory-only: no human result or output file is made.
        with patch.object(p, 'read', side_effect=[packet, {}]), patch.object(p, 'write') as write:
            with self.assertRaises(AssertionError): p.lock_human(p.ROOT/'outputs/synthetic_nonexistent.json')
            write.assert_not_called()

    def test_incomplete_draft(self):
        self.reject(dict(stage='A', complete=False, reviewer_id='TEST_ONLY', experience='synthetic'))

    def test_ai_type(self):
        self.reject(dict(stage='A', complete=True, reviewer_id='TEST_ONLY', experience='synthetic',reviewer_type='AI'))

    def test_ai_identity(self):
        self.reject(dict(stage='A', complete=True, reviewer_id='Codex', experience='synthetic',reviewer_type='human'))

    def test_outside_authorized_root(self):
        with self.assertRaises(AssertionError): p.lock_human(p.ROOT.parent/'outside-review.json')


if __name__=='__main__': unittest.main()
