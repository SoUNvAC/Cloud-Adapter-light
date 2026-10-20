"""Package existing fixed PixBox references for offline Phase301 CPU analysis."""
import hashlib
import json
from pathlib import Path
import zipfile

R=Path(__file__).resolve().parents[2]
O=R/'outputs/phase301/phase301_20261020/deployment'


def main():
    O.mkdir(parents=True,exist_ok=False)
    reference=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    content=reference.read_bytes()
    digest=hashlib.sha256(content).hexdigest()
    assert digest=='b1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'
    masks=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip'
    lock=dict(PixBox_reference_sha256=digest,reference_bytes=len(content),
              Catalogue_masks_sha256=hashlib.sha256(masks.read_bytes()).hexdigest(),
              source='Existing Phase300 local references; internal transfer only; no new annotations/downloads',
              neural_calls=0)
    with zipfile.ZipFile(O/'inputs.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('deduplicated_locations.json',content)
        z.writestr('INPUT_LOCK.json',json.dumps(lock,indent=2)+'\n')
    archive=O/'inputs.zip'
    (O/'BUNDLE_RECEIPT.json').write_text(json.dumps(dict(bytes=archive.stat().st_size,
       sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),**lock),indent=2)+'\n')
    print(json.dumps(dict(bytes=archive.stat().st_size,reference_sha256=digest)))


if __name__=='__main__': main()
