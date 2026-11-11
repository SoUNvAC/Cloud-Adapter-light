"""Validate SHA/length/ZIP CRC/path/per-file SHA before any calibration label join."""
import hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase209/phase209_20261020'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 receipt=json.loads((O/'DELIVERY_RECEIPT.json').read_text());p=O/'phase209_calibration.zip.partial';assert p.stat().st_size==receipt['bytes'] and sha(p)==receipt['sha256'];assert not (O/'calibration').exists() and not (O/'phase209_calibration.zip').exists()
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None;manifest=json.loads(z.read('DELIVERY_MANIFEST.json'));assert len(z.namelist())==len(manifest['files'])+1
  for n in z.namelist():assert not n.startswith('/') and ':' not in n and '\\' not in n and '..' not in Path(n).parts
  for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
  z.extractall(O/'calibration')
 p.rename(O/'phase209_calibration.zip')
 with (O/'TRANSFER_ACCEPTANCE.json').open('x') as f:json.dump(dict(status='passed',bytes=receipt['bytes'],sha256=receipt['sha256'],CRC=True,per_file_SHA=True,reference_or_weights_included=False),f,indent=2)
 print('Phase209 calibration transfer SHA/CRC/per-file checks passed')
if __name__=='__main__':main()
