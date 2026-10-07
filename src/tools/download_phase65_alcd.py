"""Pinned ALCD release intake; reuse tested range download, never extract pixels."""
import gzip
import tarfile
from pathlib import PurePosixPath

import download_phase65_sentinel as shared

RECORD = '1460961'
NAME = 'SENTINEL_2_reference_cloud_masks_Baetens_Hagolle.tgz'
MD5 = 'ee035e0d22a441086cfaabcface3cf24'
base_verify = shared.verify


def verify(path, expected_size, expected_md5):
    result = base_verify(path, expected_size, expected_md5)
    # Read to gzip EOF explicitly: validates trailer CRC and uncompressed size.
    with gzip.open(path, 'rb') as stream:
        while stream.read(8 * 1024 * 1024):
            pass
    count = 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name:
                raise ValueError('Unsafe archive member path')
            if not (member.isfile() or member.isdir()):
                raise ValueError('Archive links/special members forbidden')
            if member.isfile():
                with archive.extractfile(member) as stream:
                    size = 0
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        size += len(block)
                if size != member.size:
                    raise ValueError('Incomplete TAR member')
            count += 1
    result.update(gzip_crc='passed', tar_integrity='passed', members=count)
    return result


def main():
    shared.RECORD = RECORD
    shared.PINNED = {NAME: MD5}
    shared.verify = verify
    shared.main()


if __name__ == '__main__':
    main()
