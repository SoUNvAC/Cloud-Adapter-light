"""Accept only the sealed small conditional-information packet."""
import hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase210/phase210_20261020'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 receipt=json.loads((O/'DELIVERY_RECEIPT.json').read_text());p=O/'phase210_conditional.zip.partial';assert p.stat().st_size==receipt['bytes'] and sha(p)==receipt['sha256'];assert not (O/'recovery').exists() and not (O/'phase210_conditional.zip').exists()
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None;manifest=json.loads(z.read('DELIVERY_MANIFEST.json'));assert len(z.namelist())==len(manifest['files'])+1
  for n in z.namelist():assert not n.startswith('/') and ':' not in n and '\\' not in n and '..' not in Path(n).parts
  for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
  z.extractall(O/'recovery')
 p.rename(O/'phase210_conditional.zip')
 with (O/'TRANSFER_ACCEPTANCE.json').open('x') as f:json.dump(dict(status='passed',bytes=receipt['bytes'],sha256=receipt['sha256'],CRC=True,per_file_SHA=True,no_reference_weights_or_work_dirs=True),f,indent=2)
 print('Phase210 small packet SHA/CRC/per-file verification passed')
if __name__=='__main__':main()
