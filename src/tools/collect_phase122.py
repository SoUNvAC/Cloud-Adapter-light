"""Package shared-only calibration/evaluation scores; no labels or work_dirs."""
import zipfile
from phase122_common import *
def main():
    assert not (ROOT/'phase122_delivery.zip').exists()
    setup();assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    files=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix=='.json' or ('scores' in p.parts and p.suffix=='.npz'))]
    assert all(p.is_relative_to(ROOT) and p.suffix not in ('.safetensors','.pth') for p in files)
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    write(ROOT/'DELIVERY_MANIFEST.json',dict(files=manifest,source='shared only',no_work_dirs=True,no_reference_labels=True))
    path=ROOT/'phase122_delivery.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'DELIVERY_MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=4000000000
    write(ROOT/'DELIVERY_RECEIPT.json',dict(bytes=path.stat().st_size,sha256=sha(path),files=len(files)+1,
        decoder=dict(bytes=(ROOT/'C_decoder_final.safetensors').stat().st_size,sha256=sha(ROOT/'C_decoder_final.safetensors'))))
    print(load(ROOT/'DELIVERY_RECEIPT.json'))
if __name__=='__main__':main()
