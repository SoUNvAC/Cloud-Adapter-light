"""Validate the small shared text archive before exposing local analysis files."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase213/phase213_20261021'
def main():
    receipt=json.loads((OUT/'DELIVERY_RECEIPT.json').read_text(encoding='utf-8-sig'));p=OUT/'phase213_text_delivery.zip'
    assert p.stat().st_size==receipt['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==receipt['sha256']
    dest=OUT/'remote_receipt';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None and all(not Path(n).is_absolute() and '..' not in Path(n).parts and not n.lower().endswith(('.pth','.npy','.npz','.png','.tif')) for n in z.namelist())
        manifest=json.loads(z.read('TEXT_DELIVERY_MANIFEST.json'))
        assert len(z.namelist())==len(manifest)+1
        for n,r in manifest.items():
            b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        z.extractall(dest)
    local=OUT/'delivery';local.mkdir(exist_ok=False)
    for p in (dest/'evaluation').iterdir():
        if p.is_file():shutil.copy2(p,local/p.name)
    shutil.copy2(dest/'EXIT_CODE',local/'EXIT_CODE')
    result={'status':'passed','archive_sha256':receipt['sha256'],'archive_CRC_all_members_passed':True,'all_manifest_file_SHA256_passed':True,'files':len(manifest),'no_weights_or_pixels_received':True}
    (OUT/'RECEIVE_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
