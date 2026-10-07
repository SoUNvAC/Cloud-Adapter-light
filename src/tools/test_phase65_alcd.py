import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from download_phase65_alcd import verify


class ArchiveTests(unittest.TestCase):
    def test_integrity_and_unsafe_member(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
            for name, safe in [('product/Classification/used_parameters.json', True), ('../escape', False)]:
                path = Path(directory) / ('safe.tgz' if safe else 'unsafe.tgz')
                with tarfile.open(path, 'w:gz') as archive:
                    member = tarfile.TarInfo(name)
                    member.size = 2
                    archive.addfile(member, io.BytesIO(b'{}'))
                checksum = hashlib.md5(path.read_bytes()).hexdigest()
                if safe:
                    self.assertEqual(verify(path, path.stat().st_size, checksum)['gzip_crc'], 'passed')
                else:
                    with self.assertRaises(ValueError):
                        verify(path, path.stat().st_size, checksum)

    def test_corrupt_gzip_trailer(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
            path = Path(directory) / 'corrupt.tgz'
            with tarfile.open(path, 'w:gz') as archive:
                archive.addfile(tarfile.TarInfo('empty'))
            payload = bytearray(path.read_bytes())
            payload[-8] ^= 1
            path.write_bytes(payload)
            with self.assertRaises(OSError):
                verify(path, len(payload), hashlib.md5(payload).hexdigest())


if __name__ == '__main__':
    unittest.main()
