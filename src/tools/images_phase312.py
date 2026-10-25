"""Unscaled six-band TOA, image-only audit guard and strict manifest ZIP members."""
import io
import sys
import zipfile
import numpy as np
from phase312_common import *

def guard(stage,rows):
    allowed={r['image_member'] for r in rows};archive=Path(rows[0]['image_archive']).resolve()
    stream=(ROOT/(stage+'_ACCESS.jsonl')).open('a',encoding='utf-8')
    def hook(event,args):
        if event=='import' and args[0].split('.')[0]=='torch':raise PermissionError('Phase312 neural imports forbidden')
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=str(args[0]).replace('\\','/').lower()
        denied=any(k in p for k in ('masks.zip','alt_masks','classification_tags','phase306','phase307','pixbox','cloudsen12_high_l1c/ann'))
        if stage in ('fit','predict') and '/evaluation/lora_' in p:denied=True
        if p.endswith('.zip') and Path(args[0]).resolve()!=archive:denied=True
        if any(k in p for k in ('/shared/','/data/','/src/tools/')):
            stream.write(json.dumps(dict(path=p,denied=denied,stage=stage))+'\n');stream.flush()
        if denied:raise PermissionError('Forbidden image-only '+stage+' access: '+p)
    sys.addaudithook(hook);old=zipfile.ZipFile.open
    def member(self,name,*args,**kwargs):
        n=name.filename if isinstance(name,zipfile.ZipInfo) else name
        if Path(self.filename).resolve()==archive:
            assert n in allowed,'Unlisted input image: '+n
            stream.write(json.dumps(dict(member=n,stage=stage))+'\n');stream.flush()
        return old(self,name,*args,**kwargs)
    zipfile.ZipFile.open=member

def image(row,stage):
    with zipfile.ZipFile(row['image_archive']) as z:raw=z.read(row['image_member'])
    assert hashlib.sha256(raw).hexdigest()==row['image_member_sha256']
    cube=np.load(io.BytesIO(raw),allow_pickle=False);assert cube.shape==(1022,1022,13) and cube.dtype==np.float32
    six=cube[...,list(INDICES)].copy();ok=np.isfinite(six)&(six!=0)&(six!=np.float32(65535/10000));valid=ok.all(-1);rgb=ok[...,:3].all(-1)
    append(stage+'_IMAGE_ACCESS.jsonl',dict(product=row['product'],member_SHA=row['image_member_sha256'],bands=BANDS,indices=INDICES,unscaled_TOA=True,RGB_valid=int(rgb.sum()),V6=int(valid.sum()),extra_excluded_RGB=int((rgb&~valid).sum()),per_band_invalid=[int((~ok[...,i]).sum()) for i in range(6)]))
    assert not (valid&~rgb).any();return six,rgb,valid

def source(s,valid):
    assert s.shape==(3,1022,1022) and np.isfinite(s).all() and (s>=0).all()
    total=s.sum(0);q=np.divide(s,total[None],out=np.zeros_like(s),where=total[None]>1e-12);pred=s.argmax(0).astype(np.uint8)
    A=valid&(total>1e-12)&(q.max(0)>=.9);P=valid&(total>1e-12)&(pred==0)&(q.max(0)<.9)
    assert not (A&P).any();return A,P,pred,q[0]
