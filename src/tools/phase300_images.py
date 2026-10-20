"""Phase300 image-only Catalogue input. No historical helper imports."""
import hashlib
import io
import json
import zipfile
from pathlib import Path


def load_image(row, access_log=None):
    import numpy as np
    archive = Path(row['image_archive'])
    with zipfile.ZipFile(archive) as source:
        content = source.read(row['image_member'])
    actual = hashlib.sha256(content).hexdigest()
    if actual != row['image_member_sha256']:
        raise ValueError('image member SHA mismatch: ' + row['product'])
    raw = np.load(io.BytesIO(content), allow_pickle=False)
    if raw.shape != tuple(row['shape']) or raw.dtype != np.float32:
        raise ValueError('unexpected image shape/dtype')
    image = raw[..., [3, 2, 8]].transpose(2, 0, 1).copy()
    good = np.isfinite(image) & (image != 0) & (image != np.float32(65535 / 10000))
    valid = good.all(axis=0)
    image[~good] = 0
    if access_log is not None:
        with Path(access_log).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(archive=str(archive), member=row['image_member'],
                                         sha256=actual, image_valid_pixels=int(valid.sum()))) + '\n')
    return image, valid


def training_access_guard(allowed_archive, log_path):
    """Deny Python file opens of any data archive other than the fit image ZIP.

    Install in the dedicated image acceptance/teacher/training subprocess only.
    This does not claim to intercept native library syscalls.
    """
    import sys
    allowed = Path(allowed_archive).resolve()
    log = Path(log_path).open('a', encoding='utf-8')

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        name = str(args[0])
        lower = name.lower()
        forbidden = ('masks.zip', 'alt_masks.zip', 'classification_tags',
                     'coordinate_metadata', 'pixbox', 'ground_truth')
        denied = any(token in lower for token in forbidden)
        if lower.endswith('.zip') and Path(name).resolve() != allowed:
            denied = True
        log.write(json.dumps(dict(event=event, path=name, denied=denied)) + '\n')
        log.flush()
        if denied:
            raise PermissionError('Phase300 image-only process denied: ' + name)

    sys.addaudithook(audit)
    return log
