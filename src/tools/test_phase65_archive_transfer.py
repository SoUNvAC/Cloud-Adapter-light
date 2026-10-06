"""Verify resume bytes and safe failure behavior without using the network."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import receive_phase65_archive as transfer


class TransferTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[2] / 'outputs/phase65/backup'
        base.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.directory.cleanup)
        self.partial = Path(self.directory.name) / 'test.partial'
        self.partial.write_bytes(b'ab')

    def process(self, content, code=0):
        return Mock(stdout=io.BytesIO(content), wait=Mock(return_value=code), poll=Mock(return_value=code))

    def answers(self):
        return ['6', hashlib.sha256(b'ab').hexdigest() + '  remote']

    def test_segmented_resume_exact_bytes(self):
        with patch.object(transfer, 'CHUNK_BYTES', 2), patch.object(transfer.subprocess, 'check_output', side_effect=self.answers()), patch.object(transfer.subprocess, 'Popen', side_effect=[self.process(b'cd'), self.process(b'ef')]) as spawn:
            transfer.receive(self.partial)
        self.assertEqual(self.partial.read_bytes(), b'abcdef')
        self.assertIn('skip=4 count=2', spawn.call_args_list[1].args[0][-1])

    def test_prefix_mismatch_never_appends(self):
        with patch.object(transfer.subprocess, 'check_output', side_effect=['6', '0' * 64]), patch.object(transfer.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                transfer.receive(self.partial)
        spawn.assert_not_called()
        self.assertEqual(self.partial.read_bytes(), b'ab')

    def test_interruption_preserves_received_prefix(self):
        with patch.object(transfer.subprocess, 'check_output', side_effect=self.answers()), patch.object(transfer.subprocess, 'Popen', return_value=self.process(b'c', 255)):
            with self.assertRaises(RuntimeError):
                transfer.receive(self.partial)
        self.assertEqual(self.partial.read_bytes(), b'abc')


if __name__ == '__main__':
    unittest.main()
