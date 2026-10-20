"""Pure-image Catalogue RGB/BGR input and fixed nine-window geometry."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

STARTS = (0, 256, 510)
WINDOWS = [(y, x) for y in STARTS for x in STARTS]


def rgb(row, log=None):
    with zipfile.ZipFile(row['image_archive']) as z:
        content = z.read(row['image_member'])
    assert hashlib.sha256(content).hexdigest() == row['image_member_sha256']
    raw = np.load(io.BytesIO(content), allow_pickle=False)
    assert raw.shape == (1022,1022,13) and raw.dtype == np.float32
    image = raw[..., [3,2,1]].transpose(2,0,1).copy()
    valid = (np.isfinite(image) & (image != 0) & (image != np.float32(65535/10000))).all(0)
    # Invalid RGB vectors are zero before scaling; retain all finite valid >1 TOA values.
    image[:, ~valid] = 0
    image *= np.float32(255)
    # Source expects BGR, then its preprocessor exchanges back to RGB exactly once.
    bgr = image[::-1].copy()
    if log:
        with Path(log).open('a') as f:
            f.write(json.dumps(dict(product=row['product'],member=row['image_member'],
                sha256=row['image_member_sha256'],valid=int(valid.sum()),
                bgr_array_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))+'\n')
    return bgr, valid


def image_guard(rows, log):
    """Audit Python opens and enforce exact fit ZIP members, not native syscalls."""
    archive = Path(rows[0]['image_archive']).resolve()
    members = {r['image_member'] for r in rows}
    assert all(Path(r['image_archive']).resolve() == archive for r in rows)
    stream = Path(log).open('a')
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str,bytes)): return
        name = str(args[0]); lower = name.lower()
        denied = any(k in lower for k in ('masks.zip','alt_masks','pixbox','ground_truth',
            'classification_tags','common_valid','eval_images.json','cloudsen12_high_l1c/img',
            'cloudsen12_high_l1c/ann'))
        if lower.endswith('.zip') and Path(name).resolve() != archive: denied = True
        if denied:
            stream.write(json.dumps(dict(path=name,denied=True))+'\n');stream.flush()
            raise PermissionError('Phase302 image-only worker: '+name)
    sys.addaudithook(hook)
    original = zipfile.ZipFile.open
    def guarded(self, name, *args, **kwargs):
        member = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if Path(self.filename).resolve() == archive:
            assert member in members, 'Unlisted image member: '+member
            stream.write(json.dumps(dict(member=member,denied=False))+'\n');stream.flush()
        return original(self,name,*args,**kwargs)
    zipfile.ZipFile.open = guarded
    return stream
