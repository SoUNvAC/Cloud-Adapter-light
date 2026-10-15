"""Hash/allowlist validation for shared aggregate delivery and private scores."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

R=Path(__file__).resolve().parents[2]
U=R/'outputs/phase119/phase119_20261015'
D=R/'data/pixbox_s2_cmix_5036991/phase119_20261015'
O=R/'src/research_results/phase119_20261015'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
def safe(name):
    p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and '\\' not in name


def main():
    r=load(U/'DELIVERY_RECEIPT.json')
    for name,key in [('point_scores_private.zip','private_archive'),('small_delivery.zip','small_archive')]:
        p=U/name;assert p.stat().st_size==r[key+'_bytes'] and sha(p)==r[key+'_sha256']
    with zipfile.ZipFile(U/'small_delivery.zip') as z:
        assert len(z.namelist())==len(set(z.namelist()))
        manifest=json.loads(z.read('SMALL_DELIVERY_SHA256.json'))
        assert set(z.namelist())==set(manifest)|{'SMALL_DELIVERY_SHA256.json'}
        for name in z.namelist():
            safe(name);assert name.endswith(('.json','.csv'))
            b=z.read(name)
            if name in manifest:assert hashlib.sha256(b).hexdigest()==manifest[name]
            save(U/'delivery'/name,b);save(O/'remote_delivery'/name,b)
    expected={}
    for p in ('E','R','N'):
        seal=load(U/'delivery'/p/'SCORES_SEALED.json')['scores_sha256']
        for key,h in seal.items():
            if key.startswith('PixBox/'):expected[p+'/scores/'+key+'/UNLABELLED_POINT_SCORES.json']=h
    assert len(expected)==35
    with zipfile.ZipFile(U/'point_scores_private.zip') as z:
        assert len(z.namelist())==35 and set(z.namelist())==set(expected)
        for name in z.namelist():
            safe(name);b=z.read(name);assert hashlib.sha256(b).hexdigest()==expected[name]
            save(D/name,b)
    save(O/'remote_delivery/DELIVERY_RECEIPT.json',(U/'DELIVERY_RECEIPT.json').read_bytes())
    print({'status':'passed','private_point_files':35,'small_files':len(manifest),'Catalogue_dense_arrays_transferred':False})


if __name__=='__main__':main()
