"""Offline fixtures test corruption detection and safe range resume."""
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import download_phase65_sentinel as downloader
from audit_phase65_sentinel_catalogue import archive_names, validate_status


class Response(io.BytesIO):
    def __init__(self, data, status=206, content_range='bytes 3-5/6'):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': content_range}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_official_hash_and_size(self):
        path = self.root / 'file.csv'
        path.write_bytes(b'abcdef')
        result = downloader.verify(path, 6, hashlib.md5(b'abcdef').hexdigest())
        self.assertEqual(result['sha256'], hashlib.sha256(b'abcdef').hexdigest())
        with self.assertRaises(ValueError):
            downloader.verify(path, 7, hashlib.md5(b'abcdef').hexdigest())
        with self.assertRaises(ValueError):
            downloader.verify(path, 6, '0' * 32)

    def test_zip_integrity(self):
        path = self.root / 'tiny.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('fixture.txt', b'fixture only')
        checksum = hashlib.md5(path.read_bytes()).hexdigest()
        self.assertEqual(downloader.verify(path, path.stat().st_size, checksum)['zip_crc'], 'passed')

    def test_safe_resume(self):
        path = self.root / 'file.csv'
        path.with_suffix('.csv.partial').write_bytes(b'abc')
        expected = hashlib.md5(b'abcdef').hexdigest()
        with patch.dict(downloader.PINNED, {'file.csv': expected}), \
                patch.object(downloader, 'request', return_value=Response(b'def')) as network:
            downloader.transfer('https://example.test/file.csv', path, 6, {}, self.root / 'status.json')
        self.assertEqual(path.read_bytes(), b'abcdef')
        self.assertEqual(network.call_args.args[1]['Range'], 'bytes=3-')

    def test_nonrange_response_preserves_partial(self):
        path = self.root / 'file.csv'
        partial = path.with_suffix('.csv.partial')
        partial.write_bytes(b'abc')
        with patch.object(downloader, 'request', return_value=Response(b'abcdef', 200)):
            with self.assertRaises(ValueError):
                downloader.transfer('https://example.test/file.csv', path, 6, {}, self.root / 'status.json')
        self.assertEqual(partial.read_bytes(), b'abc')
        self.assertFalse(path.exists())

    def test_incomplete_download_not_training_ready(self):
        with self.assertRaises(ValueError):
            validate_status(dict(record='4172871', status='downloading', verified_files={}))
        with self.assertRaises(ValueError):
            validate_status(dict(record='4172871', status='all_files_verified', verified_files={}))

    def test_zip_path_traversal_rejected(self):
        path = self.root / 'unsafe.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('../outside.npy', b'fixture only')
        with self.assertRaises(ValueError):
            archive_names(path, '.npy')


if __name__ == '__main__':
    unittest.main()
